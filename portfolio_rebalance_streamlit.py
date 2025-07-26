import streamlit as st
import numpy as np
import yfinance as yf
import pandas as pd
import seaborn as sns
import os
from numbers_parser import Document, NegativeNumberStyle # <-- 加入 NegativeNumberStyle
import plotly.graph_objects as go
from datetime import date
from dateutil.relativedelta import relativedelta # 需要 pip install python-dateutil

# dataframe 中英文對齊設定
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
# --- 全域設定 ---
# 支援使用者直接輸入的貨幣種類
SUPPORTED_CURRENCIES_INPUT = ['TWD', 'USD', 'JPY']
# 所有計算的基準貨幣
BASE_CURRENCY = 'USD'

sns.set_theme()

def load_data_from_numbers(filename="portfolio_tracker.numbers"):
    """從 Apple Numbers 檔案中讀取投資組合數據。"""
    st.spinner(f"正在從 '{filename}' 讀取數據...")
    try:
        doc = Document(filename)
        table = doc.sheets[0].tables[0]
        table_data = table.rows(values_only=True)
        df = pd.DataFrame(table_data[1:], columns=table_data[0])
        df = df[df['Ticker'].notna() & (df['Ticker'] != '')].copy()

        # --- 欄位驗證 ---
        required_columns = ['Ticker', 'Shares', 'Target ratio', 'Categories']
        if not all(col in df.columns for col in required_columns):
            raise KeyError(f"Numbers 檔案缺少必要的欄位。請確保包含: {required_columns}")

        # --- 新增：目標比例空值驗證 ---
        # 使用 pd.to_numeric 檢查哪些值無法轉換成數字
        numeric_check = pd.to_numeric(df['Target ratio'], errors='coerce')
        invalid_rows = df[numeric_check.isna()]

        if not invalid_rows.empty:
            problem_tickers = invalid_rows['Ticker'].tolist()
            st.write("\n" + "="*50)
            st.write("錯誤：資料讀取中止！")
            st.write(f"以下股票代號的 'Target ratio' 欄位為空或非數字，請檢查您的 Numbers 檔案：")
            for ticker in problem_tickers:
                st.write(f"- {ticker}")
            st.write("="*50)
            exit() # 中止程式
        # --- 驗證結束 ---
        
        df['Target ratio'] = df['Target ratio'].astype(float)
        portfolio_series = pd.Series(df['Target ratio'].values, index=df['Ticker'])
        if portfolio_series.sum() == 0:
            raise ValueError("所有資產的 'Target ratio' 總和為 0，無法進行正規化。請至少為一項資產設定目標比例。。")
        portfolio_series /= portfolio_series.sum()
        
        quantities_array = df['Shares'].astype(float).to_numpy()
        asset_tickers_list = df['Ticker'].tolist()

        return portfolio_series, quantities_array, asset_tickers_list, table, df, doc
        
    except Exception as e:
        st.error(f"讀取或處理 Numbers 檔案時發生錯誤: {e}")
        st.stop()

def get_asset_and_fx_data(tickers_list):
    """
    獲取所有資產的價格、貨幣資訊，以及所有需要的匯率（使用更穩健的混合模式）。
    """
    st.spinner("\n正在從 Yahoo Finance 獲取資產數據...")
    tickers_str = ' '.join(tickers_list)
    tickers = yf.Tickers(tickers_str)
    
    asset_currencies = {}
    unique_currencies = set()

    # --- 修改：使用更穩健的混合模式獲取貨幣 ---
    for ticker_symbol, ticker_obj in tickers.tickers.items():
        currency = None
        # 1. 優先根據後綴判斷，無需額外網路請求，穩定快速
        if ticker_symbol.endswith('.TW'):
            currency = 'TWD'
        # (未來可以繼續增加其他市場的判斷，例如 .T 代表 JPY)
        # elif ticker_symbol.endswith('.T'):
        #     currency = 'JPY'
        
        # 2. 如果沒有符合的後綴，再嘗試用 .info 查詢
        if currency is None:
            try:
                currency = ticker_obj.info.get('currency', BASE_CURRENCY).upper()
            except Exception:
                st.warning(f"警告：無法獲取 {ticker_symbol} 的貨幣資訊，將預設為 {BASE_CURRENCY}。")
                currency = BASE_CURRENCY
        
        asset_currencies[ticker_symbol] = currency
        unique_currencies.add(currency)
    # --- 修改結束 ---

    st.spinner(f"偵測到資產貨幣: {list(unique_currencies)}")
    
    fx_tickers_to_fetch = [f"{c}=X" for c in unique_currencies if c != BASE_CURRENCY]
    fx_rates = {BASE_CURRENCY: 1.0}
    
    if fx_tickers_to_fetch:
        st.spinner(f"正在獲取匯率: {fx_tickers_to_fetch}")
        fx_data = yf.Tickers(' '.join(fx_tickers_to_fetch))
        for fx_ticker in fx_tickers_to_fetch:
            currency_code = fx_ticker.replace("=X", "")
            try:
                rate = fx_data.tickers[fx_ticker].history(period='5d')['Close'].ffill().iloc[-1]
                if pd.isna(rate):
                    raise ValueError(f"Rate for {fx_ticker} is NaN.")
                fx_rates[currency_code] = rate
            except Exception as e:
                st.error(f"錯誤：無法獲取匯率 {fx_ticker}，程式將終止。")
                st.stop()
    
    prices = tickers.history(period='5d')['Close'].ffill().iloc[-1]
    prices = prices.reindex(tickers_list)
    #st.success("資產數據與匯率獲取完成。")
    return prices, asset_currencies, fx_rates

def get_currency_map(tickers_list: list) -> dict:
    """
    僅獲取資產列表的計價貨幣對照表。
    """
    st.spinner("正在獲取資產的貨幣資訊...")
    asset_currencies = {}
    for ticker_symbol in tickers_list:
        currency = None
        # 優先根據後綴判斷
        if ticker_symbol.endswith('.TW'):
            currency = 'TWD'
        # elif ticker_symbol.endswith('.T'): # 可為日股等增加規則
        #     currency = 'JPY'
        
        # 若無規則匹配，再嘗試 API 查詢
        if currency is None:
            try:
                currency = yf.Ticker(ticker_symbol).info.get('currency', BASE_CURRENCY).upper()
            except Exception:
                st.warning(f"警告：無法獲取 {ticker_symbol} 的貨幣資訊，將預設為 {BASE_CURRENCY}。")
                currency = BASE_CURRENCY
        
        asset_currencies[ticker_symbol] = currency
    #st.write(f"偵測到資產貨幣: {list(set(asset_currencies.values()))}")
    return asset_currencies



def rebalance(investment_base, current_values_base, portfolio, is_withdraw, sell_allowed, buy_allowed):
    """通用再平衡計算函式。"""

    if is_withdraw:
        total_asset_base = current_values_base.sum() + investment_base
        target_values = portfolio * total_asset_base
        investment_diff = target_values - current_values_base
        if buy_allowed or not any(investment_diff > 0):
            return investment_diff
        else:
            sub_portfolio = portfolio[investment_diff < 0]
            sub_value = current_values_base[investment_diff < 0]
            sub_result = rebalance(investment_base, sub_value, sub_portfolio / sub_portfolio.sum(), is_withdraw, sell_allowed, buy_allowed)
            return sub_result.reindex(portfolio.index, fill_value=0)
    else: # is_invest
        total_asset_base = current_values_base.sum() + investment_base
        target_values = portfolio * total_asset_base
        investment_diff = target_values - current_values_base
        if sell_allowed or not any(investment_diff < 0):
            return investment_diff
        else:
            sub_portfolio = portfolio[investment_diff > 0]
            sub_value = current_values_base[investment_diff > 0]
            sub_result = rebalance(investment_base, sub_value, sub_portfolio / sub_portfolio.sum(), is_withdraw, sell_allowed, buy_allowed)
            return sub_result.reindex(portfolio.index, fill_value=0)


# (這是一個全新的函式)
def rebalance_by_category(investment_base, current_values_base, portfolio, df_data, is_withdraw, sell_allowed, buy_allowed):
    """
    執行兩階段的資產類別優先再平衡。
    """
    st.spinner("執行資產類別優先的兩階段再平衡...")
    
    # --- 數據準備 ---
    # 將 category 資訊合併到 portfolio 和 current_values
    df_merged = pd.DataFrame({
        'current_value': current_values_base,
        'target_ratio': portfolio
    }).join(df_data.set_index('Ticker')['Categories'])
    df_merged['Categories'] = df_merged['Categories'].fillna("Uncategorized assets")
    # --- 第一階段：類別層級的再平衡 ---
    # 按類別分組，計算每個類別的當前總價值和目標總比例
    category_values = df_merged.groupby('Categories')['current_value'].sum()
    category_targets = df_merged.groupby('Categories')['target_ratio'].sum()
    
    # 呼叫 rebalance 函式計算每個類別需要投入/提領的金額
    category_investment_diff = rebalance(investment_base, category_values, category_targets, is_withdraw, sell_allowed, buy_allowed)
    
    # --- 第二階段：資產層級的再平衡 ---
    final_investment_diff = pd.Series(0.0, index=portfolio.index)

    for category, cat_invest_amount in category_investment_diff.items():
        if abs(cat_invest_amount) < 1e-6: # 忽略極小的金額
            continue
            
        # 篩選出該類別內的所有資產
        assets_in_category = df_merged[df_merged['Categories'] == category]
        cat_is_withdraw = cat_invest_amount < 0
        
        # 對該類別內的資產進行第二輪 rebalance
        sub_rebalance_result = rebalance(
            investment_base=cat_invest_amount,
            current_values_base=assets_in_category['current_value'],
            portfolio=assets_in_category['target_ratio'] / assets_in_category['target_ratio'].sum(),
            is_withdraw=cat_is_withdraw,
            sell_allowed=sell_allowed, 
            buy_allowed=buy_allowed
        )
        # 將結果加總到最終差異中
        final_investment_diff = final_investment_diff.add(sub_rebalance_result, fill_value=0)
        
    return final_investment_diff
def calculate_transactions(result_base, prices, asset_currencies, fx_rates):
    """根據基準貨幣的再平衡結果，計算各幣別的實際交易。"""
    result_base = result_base.round(2)
    buy_assets = result_base[result_base > 0]
    sell_assets = result_base[result_base < 0]
    
    # 處理買入金額 (換算回原始貨幣)
    buy_amounts_local = buy_assets.copy()
    for asset, amount_base in buy_assets.items():
        currency = asset_currencies[asset]
        rate = fx_rates.get(currency, 1.0)
        buy_amounts_local[asset] = amount_base * rate
    # 處理賣出股數
    sell_quantities_local = sell_assets.copy()
    for asset, amount_base in sell_assets.items():
        currency = asset_currencies[asset]
        rate = fx_rates.get(currency, 1.0)
        price_local = prices[asset]
        # 賣出價值 (原始貨幣) = 賣出價值 (基準貨幣) * 匯率
        sell_value_local = abs(amount_base) * rate
        sell_quantities_local[asset] = sell_value_local / price_local if price_local > 0 else 0

    return buy_amounts_local, sell_quantities_local



@st.fragment()
def download_rebalanced_numbers(data_to_download):
    st.download_button(
                        label="📥 點此下載包含交易建議的 Numbers 檔案",
                        data=data_to_download, # 使用從暫存檔讀取出的位元組
                        file_name="rebalanced_portfolio.numbers",
                        mime="application/octet-stream"
                    )

@st.fragment()
def create_portfolio_charts(tickers_list: list, quantities_array: np.ndarray, asset_currencies: dict, option, option_map) -> tuple[go.Figure, go.Figure]:
    """
    計算投資組合總價值與累積績效，並產生兩張對應的圖表（已加入前期數據緩衝以處理開頭缺值問題）。

    Returns:
        (go.Figure, go.Figure): 一個包含 (總價值圖, 累積績效圖) 的元組。
    """
    st.spinner("正在產生投資組合總資產走勢圖...")

    quantities_dict = dict(zip(tickers_list, quantities_array))

    # --- 改進 1: 獲取 5 年的數據作為緩衝 ---
    end_date = date.today()
    start_date_actual = end_date - relativedelta(months=option)
    
    
    try:
        # 使用 start 和 end 參數獲取指定區間數據
        #隨便塞一個 yahoo finance 有提供每日價格的東西讓美元兌美元不會只有一天的價格
        tickers_list_fake = tickers_list + ["^GSPC"]
        asset_prices_hist = yf.Tickers(' '.join(tickers_list_fake)).history(period="5y", interval="1d", back_adjust=True)['Close'].ffill().fillna(1)
        
        if asset_prices_hist.empty:
            raise ValueError("無法獲取任何資產的歷史價格。")
    except Exception as e:
        st.error(f"獲取資產歷史價格時出錯: {e}")
        return go.Figure(), go.Figure()

    twd_fx_rates = {}
    currencies_to_twd = {c for c in asset_currencies.values() if c != 'TWD'}
    if currencies_to_twd:
        fx_tickers_to_twd = [f"{c}TWD=X" for c in currencies_to_twd]
        try:
            twd_fx_hist = yf.Tickers(' '.join(fx_tickers_to_twd)).history(period="5y", interval="1d", back_adjust=True)['Close'].ffill()
            if twd_fx_hist.empty: raise ValueError("無法獲取對台幣的匯率數據。")
            for fx_ticker in fx_tickers_to_twd:
                currency_code = fx_ticker.replace("TWD=X", "")
                twd_fx_rates[currency_code] = twd_fx_hist.get(fx_ticker)
        except Exception:
            st.warning("部分匯率數據獲取失敗，可能影響總值計算。")

    daily_values_twd = pd.DataFrame(index=asset_prices_hist.index)
    for ticker in tickers_list:
        if ticker not in asset_prices_hist.columns:
            st.warning(f"缺少 {ticker} 的價格數據，將從總值計算中忽略。")
            continue
            
        native_currency = asset_currencies.get(ticker, 'USD')
        prices_native = asset_prices_hist[ticker]
        daily_value_native = prices_native * quantities_dict[ticker]
        
        if native_currency == 'TWD':
            daily_values_twd[ticker] = daily_value_native
        elif native_currency in twd_fx_rates and twd_fx_rates[native_currency] is not None:
            fx_rate_series = twd_fx_rates[native_currency]
            temp_df = pd.concat([daily_value_native.rename('value'), fx_rate_series.rename('rate')], axis=1).ffill()
            daily_values_twd[ticker] = temp_df['value'] * temp_df['rate']
        else:
            daily_values_twd[ticker] = 0
            
            
    total_portfolio_value = daily_values_twd.sum(axis=1).dropna()

    if total_portfolio_value.empty:
        st.error("計算總資產價值失敗，可能是由於數據不足。")
        return go.Figure(), go.Figure()

    # --- 改進 2: 計算完成後，將數據裁切回選取期間 ---
    total_portfolio_value_oneyear = total_portfolio_value[total_portfolio_value.index.date >= start_date_actual]

    # --- 圖表一：總資產價值 (TWD) ---
    fig_value = go.Figure()
    gradient_start = total_portfolio_value_oneyear.max()
    gradient_stop = total_portfolio_value_oneyear.min()
    fig_value.add_trace(go.Scatter(
        x=total_portfolio_value_oneyear.index, y=total_portfolio_value_oneyear,
        mode='lines', name='總資產', line=dict(color='deepskyblue', width=2), fill='tozeroy',
        fillgradient=dict(colorscale=[(0.0, "rgba(29,66,131,0)"), (0.5,  "rgba(29,66,131,0.5)"), (1.0,  "rgba(29,66,131,1)")], type='vertical', start=gradient_start, stop=gradient_stop), showlegend=False))
    y_min = total_portfolio_value_oneyear.min() * 0.98
    y_max = total_portfolio_value_oneyear.max() * 1.02
    fig_value.update_layout(
        title=f'投資組合近{option_map[option]}總資產走勢 (以台幣計價)',
        yaxis_title='總資產價值 (TWD)', xaxis_title='日期',
        template='plotly_dark', height=500, yaxis_tickformat=',.0f',
        yaxis=dict(range=[y_min, y_max]), hovermode=None
    )
    fig_value.update_xaxes(showspikes=True, spikecolor="gray", spikesnap="cursor", spikemode="across+marker", spikedash="dot", spikethickness=0.5)
    fig_value.update_yaxes(showspikes=True, spikecolor="gray", spikethickness=0.5, spikedash="dot", spikemode="across+marker")
    fig_value.update_traces(hovertemplate='    %{x}<br>'+
                            '    NT$%{y}')
    # --- 圖表二：累積績效 (%) ---
    # --- 改進 3: 使用一年前的數據作為績效計算的起點 ---
    if not total_portfolio_value_oneyear.empty:
        start_value = total_portfolio_value_oneyear.iloc[0]
        # 績效仍然在完整的序列上計算，以確保平滑，然後再裁切
        performance_pct = (total_portfolio_value / (start_value+1) - 1) * 100 
        performance_pct_oneyear = performance_pct[performance_pct.index.date >= start_date_actual]
    else:
        performance_pct_oneyear = pd.Series() # 創建空的 Series 避免錯誤

    fig_perf = go.Figure()
    if not performance_pct_oneyear.empty:
        threshold = 0
        color_key = 'lightcoral'
        if performance_pct_oneyear[-1] < threshold:
            color_key = 'lightgreen'
        gradient_start_stop = performance_pct_oneyear.abs().max()*0.5
        fig_perf.add_trace(go.Scatter(x=performance_pct_oneyear.index, y=performance_pct_oneyear,
            mode='lines', name='累積績效', line=dict(color=color_key, width=2),
            fill='tozeroy', fillgradient=dict(colorscale='rdylgn', type='vertical', start=gradient_start_stop, stop=-gradient_start_stop),
                                              showlegend=False))
    fig_perf.update_layout(
        title='投資組合累積績效 (%)',
        yaxis_title='績效 (%)', xaxis_title='日期',
        template='plotly_dark', height=500,
        yaxis_ticksuffix=' %', hovermode=None
    )
    fig_perf.update_xaxes(showspikes=True, spikecolor="white", spikesnap="cursor", spikemode="across+marker", spikedash="dot", spikethickness=1.5)
    fig_perf.update_yaxes(showspikes=True, spikecolor="white", spikethickness=1.5, spikedash="dot", spikemode="across+marker")
    fig_perf.update_traces(hovertemplate='   %{x}<br>'+
                           '   %{y:.2f}%')
    return fig_value, fig_perf



def pills(option_map):
    options = option_map.keys()
    select = st.pills('時間範圍', options = options, 
                          format_func=lambda option: option_map[option], selection_mode='single', default=1)
    return select


@st.fragment()
def charts(tickers_list, quantities, asset_currencies):
    option_map = {1: '一個月',
                      3: '三個月',
                      6: '六個月',
                      12: '一年',
                      36: '三年'}
    select = pills(option_map)
    # 將獲取的貨幣對照表傳遞給繪圖函式
    fig_value, fig_perf = create_portfolio_charts(tickers_list, quantities, asset_currencies, select, option_map)
    # --- 改進 2：使用 st.tabs 建立分頁 ---
    tab1, tab2 = st.tabs(["總資產價值 (TWD)", "累積績效 (%)"])
    with tab1:
        st.plotly_chart(fig_value, use_container_width=True)

    with tab2:
        st.plotly_chart(fig_perf, use_container_width=True)
    


    
@st.fragment
def operation_type():
    col1, col2 = st.columns(2)
    with col1:
        investment_type=st.radio("操作類型：", ('投入資金', '提領資金'))
    with col2:
        rebalance_priority = st.radio(
            "再平衡優先級：",
            ('個別資產', '資產類別優先'),
            help="選擇『資產類別優先』會啟用兩階段再平衡，確保大類別的比例優先滿足目標。"
        )
        by_category = (rebalance_priority == '資產類別優先')
    return investment_type, by_category

@st.fragment()
def sell_or_buy():
    buy_allowed, sell_allowed = False, False
    buy_allowed=st.checkbox("投入/提領時，允許賣出/買入部分資產以達成平衡？", help="若不允許，則投入時只會計算需要買入（佔比較低）的資產，\n 提領時只會計算需要賣出（佔比較高）的資產。")
    sell_allowed = buy_allowed
    return buy_allowed, sell_allowed

# (此函式用於替換舊版本)
@st.fragment()
def create_polar_comparison_charts(
    before_ratios: pd.Series, 
    after_ratios: pd.Series, 
    target_ratios: pd.Series,
    before_values_twd: pd.Series,
    after_values_twd: pd.Series,
    df_data: pd.DataFrame #<-- 新增 df 參數以獲取 category
) -> tuple[go.Figure, go.Figure, go.Figure, go.Figure]:
    """
    建立資產層級和類別層級的極座標柱狀圖。
    """
    # --- 數據準備與排序 ---
    df_merged = pd.DataFrame({
        'before_ratio': before_ratios,
        'after_ratio': after_ratios,
        'target_ratio': target_ratios,
        'before_value_twd': before_values_twd,
        'after_value_twd': after_values_twd
    }).join(df_data.set_index('Ticker')[['Categories']])
    df_merged['Categories'] = df_merged['Categories'].fillna("Uncategorized assets")
    # 按類別總價值 -> 資產總價值 排序
    df_merged['cat_value'] = df_merged.groupby('Categories')['before_value_twd'].transform('sum')
    df_sorted = df_merged.sort_values(by=['cat_value', 'before_value_twd'], ascending=[False, False])
    
    # --- 顏色邏輯 ---
    # 1. 產生類別顏色
    unique_categories = df_sorted['Categories'].unique()
    category_colors = sns.color_palette('viridis_r', n_colors=len(unique_categories)).as_hex()
    cat_color_map = dict(zip(unique_categories, category_colors))

    # 2. 產生資產顏色
    asset_colors = []
    for category in unique_categories:
        assets_in_cat = df_sorted[df_sorted['Categories'] == category]
        # 為該類別的資產產生從深到淺的漸層色
        cat_base_color = cat_color_map[category]
        asset_palette = sns.light_palette(cat_base_color, n_colors=len(assets_in_cat)+3, reverse=True)
        # --- FIX: Convert RGB tuple to hex string ---
        # The original `color.hex` was incorrect because asset_palette contains RGB tuples.
        hex_palette = [f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}' for r, g, b in asset_palette]
        asset_colors.extend(hex_palette[:-3])
        # --- FIX END ---
    
    # --- 繪製圖表 ---
    # (內部輔助函式 _create_single_polar_chart 不變，但顏色參數改為傳入)
    # 內部輔助函式，用於繪製單張圖表
    def _create_single_polar_chart(
        actual_ratios: pd.Series, 
        target_ratios: pd.Series, 
        actual_values_twd: pd.Series, #<-- 新增參數
        title: str,
        colors
    ) -> go.Figure:
        # 確保數據對齊
        target_ratios = target_ratios.reindex(actual_ratios.index).fillna(0)
        actual_values_twd = actual_values_twd.reindex(actual_ratios.index).fillna(0)

        # --- 計算圖形參數 ---
        widths = target_ratios.values * 360
        thetas = np.cumsum(widths) - 0.5 * widths
        base_radius = 6
        Radius = 10        #外圈半徑
        r_values = np.sqrt(base_radius**2 + (actual_ratios.values / (target_ratios.values + 1e-9)) * (Radius**2 - base_radius**2)) - base_radius

        # --- FIX: 準備 customdata ---
        # 將台幣價值和實際比例(%)打包
        # customdata 的每一行對應一個資產，[價值, 比例]
        custom_data_stack = np.stack(
            [actual_values_twd.values, actual_ratios.values * 100, target_ratios.values * 100], 
            axis=-1
        )

        fig = go.Figure()
        

        # 新增一個代表 100% 目標的基準線環
        fig.add_trace(go.Scatterpolar(
            r=np.ones(120) * Radius,
            theta=np.linspace(0, 360, 120),
            mode='lines',
            name='目標基準',
            line_color='gray',
            line=dict(dash='dash', shape='spline', smoothing=1, width=1.5),
            hoverinfo="none"
        ))
        # 新增代表實際比例的柱狀圖
        fig.add_trace(go.Barpolar(
            r=r_values,
            theta=thetas,
            width=widths,
            marker_color=colors,
            marker_line_color="black",
            marker_line_width=2,
            text=actual_ratios.index,
            opacity=0.8,
            base=base_radius,
            customdata=custom_data_stack, #<-- 綁定 customdata
            # --- FIX: 更新 hovertemplate ---
            hovertemplate=(
                '<b>%{text}</b><br><br>'
                '目前價值: TWD$%{customdata[0]:,.0f}<br>'
                '目前比例: %{customdata[1]:.2f}%<br>'
                '目標比例: %{customdata[2]:.2f}%'
                '<extra></extra>'
            ),
            name='實際比例'
        ))


        # (美化圖表佈局的程式碼不變，此處省略)
        fig.update_layout(
            title=title,
            template='plotly_dark',
            height=600,
            dragmode = False,
            polar=dict(
                radialaxis=dict(
                    visible=False,
                    range=[0, max(Radius*1.2, (r_values.max()+base_radius) * 1.1)], # 動態調整半徑軸範圍
                    showticklabels=False, 
                    ticks=''
                ),
                angularaxis=dict(
                    showticklabels=False,
                    ticks='',
                    visible = False
                ))
        )
        return fig

    # 繪製資產層級圖表
    fig_before_asset = _create_single_polar_chart(df_sorted['before_ratio'], df_sorted['target_ratio'], df_sorted['before_value_twd'], "資產層級 (平衡前)", asset_colors)
    fig_after_asset = _create_single_polar_chart(df_sorted['after_ratio'], df_sorted['target_ratio'], df_sorted['after_value_twd'], "資產層級 (平衡後)", asset_colors)

    # 繪製類別層級圖表
    cat_before_ratios = df_sorted.groupby('Categories')['before_ratio'].sum().sort_values(ascending=False)
    cat_target_ratios = df_sorted.groupby('Categories')['target_ratio'].sum().reindex(cat_before_ratios.index)
    cat_before_values = df_sorted.groupby('Categories')['before_value_twd'].sum().reindex(cat_before_ratios.index)
    
    cat_after_ratios = df_sorted.groupby('Categories')['after_ratio'].sum().reindex(cat_before_ratios.index)
    cat_after_values = df_sorted.groupby('Categories')['after_value_twd'].sum().reindex(cat_before_ratios.index)
    
    # 顏色使用排序後的類別基礎色
    sorted_cat_colors = [cat_color_map[cat] for cat in cat_before_ratios.index]

    fig_before_cat = _create_single_polar_chart(cat_before_ratios, cat_target_ratios, cat_before_values, "類別層級 (平衡前)", sorted_cat_colors)
    fig_after_cat = _create_single_polar_chart(cat_after_ratios, cat_target_ratios, cat_after_values, "類別層級 (平衡後)", sorted_cat_colors)
    
    return fig_before_asset, fig_after_asset, fig_before_cat, fig_after_cat

# --- Streamlit 網頁應用主體 ---
def web_main():
    # 設定網頁標題和說明
    st.set_page_config(page_title="資產再平衡計算機", layout="wide")
    st.title("📈 資產組合再平衡計算機")
    st.markdown("""
    這個工具可以幫助您根據目標比例，計算出再平衡所需的交易。
    請修改並上傳您的 Apple Numbers 追蹤檔案 (`.numbers`) 來開始。
    """)

    # --- 功能 1: 提供範例檔案下載 ---
    try:
        with open("portfolio_tracker.numbers", "rb") as fp:
            st.download_button(
                label="📥 點此下載 Numbers 範本檔案",
                data=fp,
                file_name="portfolio_tracker_template.numbers",
                mime="application/octet-stream"
            )
    except FileNotFoundError:
        st.warning("警告：找不到範本檔案 'portfolio_tracker.numbers'。下載功能將無法使用。")

    st.markdown("---") # 分隔線

    # 1. 檔案上傳元件
    uploaded_file = st.file_uploader("上傳您的 portfolio_tracker.numbers 檔案", type=["numbers"])

    if uploaded_file is not None:
        # --- 修正：將上傳的檔案寫入暫存檔 ---
        # 定義一個暫存檔案的路徑
        temp_file_path = "temp_uploaded_portfolio.numbers"
        # 將使用者上傳的檔案內容寫入這個暫存路徑
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        # --- 修正結束 ---

        # 讀取檔案並進行初步驗證
        try:
            # 現在傳遞的是暫存檔案的路徑(字串)，而不是記憶體物件
            portfolio, quantities, tickers_list, table, df, doc = load_data_from_numbers(temp_file_path)
            st.success("Numbers 檔案讀取成功！")
            st.dataframe(df) # 在網頁上顯示讀取到的表格
        except Exception as e:
            st.error(f"讀取 Numbers 檔案時出錯：{e}")
            # 清理暫存檔 (可選)
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            st.stop() # 出錯則停止執行

        # --- 重構後的數據獲取流程 ---
        # 2.A. 獲取所有資產的貨幣對照表
        asset_currencies = get_currency_map(tickers_list)
        
        # 2.B. 獲取「最新」的價格和匯率，僅用於「再平衡計算」
        st.spinner("正在獲取最新價格與匯率用於計算...")
        unique_currencies = set(asset_currencies.values())
        # --- FIX: 強制加入 'TWD' 到匯率獲取清單 ---
        # 因為圖表最終需要以 TWD 呈現，所以無論如何都要取得 TWD 匯率
        unique_currencies.add('TWD')
        # --- 修正結束 ---
        fx_tickers_to_fetch = [f"{c}=X" for c in unique_currencies if c != BASE_CURRENCY]
        
        all_tickers_for_latest_price = tickers_list + fx_tickers_to_fetch
        latest_data = yf.Tickers(' '.join(all_tickers_for_latest_price)).history(period="5d", interval="1d")['Close'].ffill()
        
        if latest_data.empty:
            st.error("無法獲取最新的市場數據，無法繼續計算。")
            st.stop()
            
        latest_prices = latest_data.iloc[-1]
        prices = latest_prices[tickers_list]
        
        fx_rates = {BASE_CURRENCY: 1.0}
        for fx_ticker in fx_tickers_to_fetch:
            currency_code = fx_ticker.replace("=X", "")
            fx_rates[currency_code] = latest_prices.get(fx_ticker)
        st.subheader("--- 總資產走勢圖 ---")
        charts(tickers_list, quantities, asset_currencies)

        # 2. 互動式輸入元件
        st.header("設定再平衡參數")

        col1, col2 = st.columns(2)
        with col1:
            investment_type, by_category = operation_type()
        
        is_withdraw = (investment_type == '提領資金')

        with col2:
            buy_allowed, sell_allowed = sell_or_buy()

        st.subheader("投入/提領金額")
        twd_invest, usd_invest, jpy_invest = 0, 0, 0
        factor = -1 if is_withdraw else 1
        with st.form(key='investment_form'):
            if "TWD" in fx_rates.keys():
                twd_invest_abs = st.number_input("台幣 (TWD)", value=0, min_value=0, format="%d")
                twd_invest = twd_invest_abs * factor
            if "USD" in fx_rates.keys():
                usd_invest_abs = st.number_input("美金 (USD)", value=0.00, min_value=0.0, format="%.2f")
                usd_invest = usd_invest_abs * factor
            if "JPY" in fx_rates.keys():
                jpy_invest_abs = st.number_input("日圓 (JPY)", value=0, min_value=0, format="%d")
                jpy_invest = jpy_invest_abs * factor
            
            

            submitted = st.form_submit_button("🚀 開始計算再平衡！", use_container_width=True)

        # 3. 執行按鈕
        if submitted:
            with st.spinner("正在獲取市場數據並執行計算..."):
                try:
                    # --- 執行核心邏輯 ---
                    investment_base = (twd_invest / fx_rates.get('TWD', 1)) + \
                                      (usd_invest / fx_rates.get('USD', 1)) + \
                                      (jpy_invest / fx_rates.get('JPY', 1))
                    
                    current_values_base = pd.Series(prices.values * quantities, index=prices.index)
                    for asset, value in current_values_base.items():
                        currency = asset_currencies.get(asset, BASE_CURRENCY)
                        if currency != BASE_CURRENCY:
                            current_values_base[asset] /= fx_rates.get(currency, 1.0)
                    
                    if is_withdraw:
                        total_withdrawal_base = abs(investment_base)
                        total_assets_base = current_values_base.sum()
                        if total_withdrawal_base > total_assets_base:
                            st.error(f"錯誤：欲提領金額 (約 ${total_withdrawal_base:,.2f}) 已超出資產總額 (約 ${total_assets_base:,.2f})。")
                            st.stop()
                    # --- 核心邏輯修改：根據使用者選擇呼叫不同的 rebalance 函式 ---
                    if by_category:
                        result_base = rebalance_by_category(investment_base, current_values_base, portfolio, df, is_withdraw, sell_allowed, buy_allowed)
                    else:
                        st.spinner("執行個別資產的單層再平衡...")
                        result_base = rebalance(investment_base, current_values_base, portfolio, is_withdraw, sell_allowed, buy_allowed)
                    buy_amounts_local, sell_quantities_local = calculate_transactions(result_base, prices, asset_currencies, fx_rates)
                    
                    # --- 在網頁上顯示結果 ---
                    st.header("📊 計算結果")
                    st.subheader("--- 交易建議 ---")
                    
                    if buy_amounts_local.empty and sell_quantities_local.empty:
                        st.info("無需進行任何交易。")
                    else:
                        col1, col2 = st.columns(2)
                        buy_df = None # 初始化
                        
                        with col1:
                            if not buy_amounts_local.empty:
                                buy_df = pd.DataFrame({'Amount_Local': buy_amounts_local})
                                aligned_prices = prices.reindex(buy_df.index)
                                buy_df['Shares_to_Buy'] = buy_df['Amount_Local'] / aligned_prices
                                buy_df['Formatted_Amount'] = buy_df.apply(
                                    lambda row: f"{asset_currencies[row.name]} {row['Amount_Local']:,.2f}",
                                    axis=1
                                )
                                display_buy_df = buy_df[['Formatted_Amount', 'Shares_to_Buy']].rename(columns={'Formatted_Amount': '買入金額', 'Shares_to_Buy': '建議股數'})
                                
                                st.write("請買入：")
                                st.dataframe(display_buy_df.round(5))
                            else:
                                st.write("請買入：")
                                st.info("無")
                    
                        with col2:
                            # --- 修正：簡化賣出建議的顯示邏輯 ---
                            if not sell_quantities_local.empty:
                                # 直接將已算好的「賣出股數」Series 轉成 DataFrame
                                sell_df = pd.DataFrame(sell_quantities_local)
                                sell_df.columns = ['建議賣出股數'] # 重新命名欄位
                                
                                st.write("請賣出：")
                                st.dataframe(sell_df.round(5))
                            else:
                                st.write("請賣出：")
                                st.info("無")
                            # --- 修正結束 ---
                    


                    
                    st.subheader("--- 圖表分析 ---")
                    before_ratio = current_values_base / current_values_base.sum()
                    adjusted_values_base = current_values_base + result_base
                    adjusted_values_base[adjusted_values_base < 0] = 0
                    after_ratio = adjusted_values_base / adjusted_values_base.sum() if adjusted_values_base.sum() > 0 else before_ratio
                    before_values_twd = current_values_base*fx_rates.get("TWD", 1.0)
                    after_values_twd = adjusted_values_base*fx_rates.get("TWD", 1.0)
                    

                    # 呼叫新的繪圖函式，它會一次返回四張圖
                    fig_before_asset, fig_after_asset, fig_before_cat, fig_after_cat = create_polar_comparison_charts(
                        before_ratios=before_ratio,
                        after_ratios=after_ratio,
                        target_ratios=portfolio,
                        before_values_twd=before_values_twd,
                        after_values_twd=after_values_twd,
                        df_data=df # 傳入 df
                    )

                    # --- UI 修改：使用 tabs 來顯示不同層級的圖表 ---
                    tab_asset, tab_category = st.tabs(["按資產顯示", "按類別顯示"])

                    with tab_asset:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.plotly_chart(fig_before_asset, use_container_width=True, theme=None)
                        with col2:
                            st.plotly_chart(fig_after_asset, use_container_width=True, theme=None)
                    
                    with tab_category:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.plotly_chart(fig_before_cat, use_container_width=True, theme=None)
                        with col2:
                            st.plotly_chart(fig_after_cat, use_container_width=True, theme=None)





                    # --- 功能 2: 產生並下載結果檔 (已加入儲存格格式化) ---
                    st.subheader("--- 下載更新後的檔案 ---")
                    
                    try:
                        shares_to_buy_col_index = df.columns.get_loc('Shares to buy')
                        # 先清空舊資料
                        for i in range(len(df)):
                            row_index = i + 1
                            table.write(row_index, shares_to_buy_col_index, 0)
                            # 順便設定預設格式
                            table.set_cell_formatting(
                                row_index, shares_to_buy_col_index, "number", decimal_places=5
                            )

                        # 寫入買入建議並設定格式
                        if 'buy_df' in locals() and buy_df is not None:
                            for ticker, row_data in buy_df.iterrows():
                                row_index = df[df['Ticker'] == ticker].index[0] + 1
                                shares_value = row_data['Shares_to_Buy']
                                table.write(row_index, shares_to_buy_col_index, shares_value)
                                table.set_cell_formatting(
                                    row_index, shares_to_buy_col_index, "number", decimal_places=5
                                )
                        
                        # 寫入賣出建議並設定格式 (以負數表示)
                        if 'sell_df' in locals() and sell_df is not None:
                            for ticker, row_data in sell_df.iterrows():
                                row_index = df[df['Ticker'] == ticker].index[0] + 1
                                shares_value = -row_data['建議賣出股數'] # 賣出寫為負數
                                table.write(row_index, shares_to_buy_col_index, shares_value)
                                table.set_cell_formatting(
                                    row_index, shares_to_buy_col_index, "number", 
                                    decimal_places=5,
                                    negative_style=NegativeNumberStyle.RED # 讓負數顯示為紅色
                                )
                    except KeyError:
                        st.warning("警告：Numbers 檔案中未找到 'Shares to buy' 欄位，無法將建議寫回檔案。")

                    # --- 核心修正處 ---
                    # 1. 定義一個新的暫存檔路徑，用於儲存結果
                    output_temp_path = "temp_rebalanced_output.numbers"
                    
                    # 2. 將修改後的 doc 物件，儲存到這個暫存檔路徑
                    doc.save(output_temp_path)

                    # 3. 從剛剛存好的暫存檔中，將內容讀取為位元組(bytes)
                    with open(output_temp_path, "rb") as f:
                        data_to_download = f.read()
                    # --- 核心修正結束 ---

                    download_rebalanced_numbers(data_to_download)
                    
                    st.success("全部流程完成！")

                except Exception as e:
                    st.error(f"計算過程中發生錯誤：{e}")

if __name__ == '__main__':
    # 為了版面整潔，再次提醒，所有函式定義都應放在 web_main() 之前
    web_main()
