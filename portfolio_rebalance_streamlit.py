import streamlit as st
import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import os
from numbers_parser import Document, NegativeNumberStyle # <-- 加入 NegativeNumberStyle
from io import BytesIO


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

        # 驗證通過後，才安全地進行型別轉換
        df['Target ratio'] = df['Target ratio'].astype(float)
        
        portfolio_series = pd.Series(df['Target ratio'].values, index=df['Ticker'])
        # 檢查目標比例總和是否為0，避免除以零的錯誤
        if portfolio_series.sum() == 0:
            st.write("\n錯誤：所有資產的 'Target ratio' 總和為 0，無法進行正規化。請至少為一項資產設定目標比例。")
            exit()
        portfolio_series /= portfolio_series.sum()
        
        quantities_array = df['Shares'].astype(float).to_numpy()
        asset_tickers_list = df['Ticker'].tolist()

        st.write("數據讀取成功！")
        return portfolio_series, quantities_array, asset_tickers_list, table, df, doc
        
    except FileNotFoundError:
        st.write(f"錯誤：找不到檔案 '{filename}'。請確保檔案與腳本在同一個資料夾中。")
        exit()
    except KeyError as e:
        st.write(f"錯誤：Numbers 檔案中缺少必要的欄位：{e}。請確認欄位名稱是否為 'Ticker', 'Shares', 'Target ratio' 等。")
        exit()
    except Exception as e:
        st.write(f"讀取或處理 Numbers 檔案時發生錯誤: {e}")
        exit()
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
    st.success("資產數據與匯率獲取完成。")
    return prices, asset_currencies, fx_rates

def get_investment_amounts(supported_currencies, fx_rates):
    """互動式詢問使用者本次投入/提領的金額，並換算成基準貨幣。"""
    def _get_numeric_input(prompt):
        while True:
            try: return float(input(prompt))
            except ValueError: st.write("無效輸入，請輸入一個數字。")

    st.write("\n請輸入本次要投入/提領的金額，若需提領請輸入負值。")
    total_investment_base_currency = 0
    for currency in supported_currencies:
        amount = _get_numeric_input(f"{currency} 金額: ")
        if currency not in fx_rates:
            st.write(f"警告: 缺少 {currency}/{BASE_CURRENCY} 匯率，此筆金額將不被計入。")
            continue
        total_investment_base_currency += amount / fx_rates[currency]
        
    if not any(c in fx_rates for c in supported_currencies):
         st.write("警告：所有輸入貨幣的匯率均無法獲取，總投入/提領金額為0。")

    return total_investment_base_currency

def get_permissions(is_withdraw):
    """根據是提領還是投入，詢問對應的權限。"""
    if is_withdraw:
        st.write("\n偵測到提領操作。")
        buy = input("提領時，是否允許買入部分資產以達成平衡？ (y/n): ").lower().strip() in ['y', 'yes']
        return False, buy # sell=False, buy=True/False
    else:
        st.write("\n偵測到投入操作。")
        sell = input("投入時，是否允許賣出部分資產以達成平衡？ (y/n): ").lower().strip() in ['y', 'yes']
        return sell, False # sell=True/False, buy=False

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

# (繪圖函式 _draw_single_donut 和 plot_rebalancing_comparison_charts 維持不變，此處省略)
def plot_rebalancing_comparison_charts(before_ratios, after_ratios, target_ratios, filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6)); fig.set_facecolor('#2d2d3d')
    _draw_single_donut(ax1, before_ratios, target_ratios, "Before Rebalancing")
    _draw_single_donut(ax2, after_ratios, target_ratios, "After Rebalancing")
    plt.tight_layout(pad=1)
    return fig
    
def _draw_single_donut(ax, current_ratios, target_ratios, title):
    target_ratios = target_ratios.reindex(current_ratios.index); categories = current_ratios.index
    current_alloc_norm, target_alloc_norm = current_ratios.values, target_ratios.values
    colors = sns.color_palette('viridis', n_colors=len(categories)).as_hex(); bg_color = "#2d2d3dff"
    ax.set_facecolor(bg_color); ax.set_title(title, color='white', fontsize=20, pad=20); ax.set_aspect('equal')
    start_angle = 90; base_radius = 1.1; center_hole_radius = 0.66
    for i in range(len(categories)):
        p_current, p_target = current_alloc_norm[i], target_alloc_norm[i]
        if p_target > 0: radius = np.sqrt(max(0, (center_hole_radius**2) + (p_current/p_target)*(base_radius**2 - center_hole_radius**2)))
        else: radius = base_radius if p_current == 0 else 0.3
        angle_slice = target_alloc_norm[i] * 360; end_angle = start_angle - angle_slice
        wedge = patches.Wedge((0,0), r=radius, theta1=end_angle, theta2=start_angle, facecolor=colors[i], edgecolor=bg_color, linewidth=2.5); ax.add_patch(wedge)
        if angle_slice > 0:
            mid_angle_rad = np.deg2rad(start_angle - angle_slice/2); text_radius = (radius + center_hole_radius)/2
            x, y = text_radius*np.cos(mid_angle_rad), text_radius*np.sin(mid_angle_rad)
            ax.text(x, y, categories[i], ha='center', va='center', color='white', fontsize=11, weight='bold', bbox=dict(boxstyle="round,pad=0.2", fc=colors[i], ec='none', alpha=0.6))
        start_angle = end_angle
    reference_circle = plt.Circle((0,0), base_radius, linestyle='--', fill=False, edgecolor='gray', linewidth=1.5); ax.add_patch(reference_circle)
    center_hole = plt.Circle((0,0), center_hole_radius, facecolor=bg_color, linestyle=''); ax.add_patch(center_hole)
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4); ax.axis('off')

@st.fragment()
def invest_withdraw():
    twd_invest = st.number_input("台幣 (TWD)", value=0)
    usd_invest = st.number_input("美金 (USD)", value=0.00, format="%.2f")
    jpy_invest = st.number_input("日圓 (JPY)", value=0)
    return twd_invest, usd_invest, jpy_invest


@st.fragment()
def download_rebalanced_numbers():
    st.download_button(
                        label="📥 點此下載包含交易建議的 Numbers 檔案",
                        data=data_to_download, # 使用從暫存檔讀取出的位元組
                        file_name="rebalanced_portfolio.numbers",
                        mime="application/octet-stream"
                    )
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

        # 2. 互動式輸入元件
        st.header("設定再平衡參數")

        col1, col2 = st.columns(2)
        with col1:
            investment_type = st.radio("操作類型：", ('投入資金', '提領資金'))
        
        is_withdraw = (investment_type == '提領資金')

        with col2:
            if is_withdraw:
                buy_allowed = st.checkbox("提領時，允許買入部分資產以達成平衡？")
                sell_allowed = False
            else:
                sell_allowed = st.checkbox("投入時，允許賣出部分資產以達成平衡？")
                buy_allowed = False

        st.subheader("投入/提領金額 (提領請輸入正數)")
        with st.form(key='investment_form'):
            twd_invest_abs = st.number_input("台幣 (TWD)", value=0, min_value=0, format="%d")
            usd_invest_abs = st.number_input("美金 (USD)", value=0.00, min_value=0.0, format="%.2f")
            jpy_invest_abs = st.number_input("日圓 (JPY)", value=0, min_value=0, format="%d")
            
            factor = -1 if is_withdraw else 1
            twd_invest = twd_invest_abs * factor
            usd_invest = usd_invest_abs * factor
            jpy_invest = jpy_invest_abs * factor

            submitted = st.form_submit_button("🚀 開始計算再平衡！", use_container_width=True)

        # 3. 執行按鈕
        if submitted:
            with st.spinner("正在獲取市場數據並執行計算..."):
                try:
                    # --- 執行核心邏輯 ---
                    prices, asset_currencies, fx_rates = get_asset_and_fx_data(tickers_list)

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
                    
                    fig = plot_rebalancing_comparison_charts(
                        before_ratios=before_ratio,
                        after_ratios=after_ratio,
                        target_ratios=portfolio,
                        filename="rebalancing_side_by_side.png" # filename is not used here but good practice
                    )
                    st.pyplot(fig)

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

                    download_rebalanced_numbers()

                    st.success("全部流程完成！")

                except Exception as e:
                    st.error(f"計算過程中發生錯誤：{e}")

if __name__ == '__main__':
    # 為了版面整潔，再次提醒，所有函式定義都應放在 web_main() 之前
    web_main()
