import streamlit as st
import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import os
from numbers_parser import Document
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
    st.text(f"正在從 '{filename}' 讀取數據...")
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
            st.text("\n" + "="*50)
            st.text("錯誤：資料讀取中止！")
            st.text(f"以下股票代號的 'Target ratio' 欄位為空或非數字，請檢查您的 Numbers 檔案：")
            for ticker in problem_tickers:
                st.text(f"- {ticker}")
            st.text("="*50)
            exit() # 中止程式
        # --- 驗證結束 ---

        # 驗證通過後，才安全地進行型別轉換
        df['Target ratio'] = df['Target ratio'].astype(float)
        
        portfolio_series = pd.Series(df['Target ratio'].values, index=df['Ticker'])
        # 檢查目標比例總和是否為0，避免除以零的錯誤
        if portfolio_series.sum() == 0:
            st.text("\n錯誤：所有資產的 'Target ratio' 總和為 0，無法進行正規化。請至少為一項資產設定目標比例。")
            exit()
        portfolio_series /= portfolio_series.sum()
        
        quantities_array = df['Shares'].astype(float).to_numpy()
        asset_tickers_list = df['Ticker'].tolist()

        st.text("數據讀取成功！")
        return portfolio_series, quantities_array, asset_tickers_list, table, df, doc
        
    except FileNotFoundError:
        st.text(f"錯誤：找不到檔案 '{filename}'。請確保檔案與腳本在同一個資料夾中。")
        exit()
    except KeyError as e:
        st.text(f"錯誤：Numbers 檔案中缺少必要的欄位：{e}。請確認欄位名稱是否為 'Ticker', 'Shares', 'Target ratio' 等。")
        exit()
    except Exception as e:
        st.text(f"讀取或處理 Numbers 檔案時發生錯誤: {e}")
        exit()
def get_asset_and_fx_data(tickers_list):
    """
    獲取所有資產的價格、貨幣資訊，以及所有需要的匯率。
    """
    st.text("\n正在從 Yahoo Finance 獲取資產數據...")
    tickers_str = ' '.join(tickers_list)
    tickers = yf.Tickers(tickers_str)
    
    asset_currencies = {}
    unique_currencies = set()

    for ticker_symbol, ticker_obj in tickers.tickers.items():
        try:
            currency = ticker_obj.info.get('currency', BASE_CURRENCY).upper()
            asset_currencies[ticker_symbol] = currency
            unique_currencies.add(currency)
        except Exception:
            st.text(f"警告：無法獲取 {ticker_symbol} 的貨幣資訊，將預設為 {BASE_CURRENCY}。")
            asset_currencies[ticker_symbol] = BASE_CURRENCY
            unique_currencies.add(BASE_CURRENCY)

    st.text(f"偵測到資產貨幣: {list(unique_currencies)}")
    
    # 獲取所有需要的匯率 (對美元)
    # --- FIX: 修正匯率代號的建構方式 ---
    # 錯誤的: f"{c}{BASE_CURRENCY}=X" (例如 TWDUSD=X)
    # 正確的: f"{c}=X" (例如 TWD=X)
    fx_tickers_to_fetch = [f"{c}=X" for c in unique_currencies if c != BASE_CURRENCY]
    fx_rates = {BASE_CURRENCY: 1.0}
    
    if fx_tickers_to_fetch:
        st.text(f"正在獲取匯率: {fx_tickers_to_fetch}")
        fx_data = yf.Tickers(' '.join(fx_tickers_to_fetch))
        for fx_ticker in fx_tickers_to_fetch:
            # --- FIX: 修正從匯率代號解析回貨幣碼的方式 ---
            currency_code = fx_ticker.replace("=X", "")
            try:
                # 使用 last-day's close price for robustness
                rate = fx_data.tickers[fx_ticker].history(period='5d')['Close'].ffill().iloc[-1]
                if pd.isna(rate):
                    raise ValueError(f"Rate for {fx_ticker} is NaN.")
                fx_rates[currency_code] = rate
            except Exception as e:
                st.text(f"錯誤：無法獲取匯率 {fx_ticker}，程式將終止。")
                st.text(f"請檢查 yfinance 是否支援此匯率代號。錯誤訊息: {e}")
                exit()
    
    # 獲取資產價格
    prices = tickers.history(period='5d')['Close'].ffill().iloc[-1]
    prices = prices.reindex(tickers_list)
    st.text("各資產最近一個交易日收盤價：")
    st.text(prices)

    st.text("資產數據與匯率獲取完成。")
    return prices, asset_currencies, fx_rates

def get_investment_amounts(supported_currencies, fx_rates):
    """互動式詢問使用者本次投入/提領的金額，並換算成基準貨幣。"""
    def _get_numeric_input(prompt):
        while True:
            try: return float(input(prompt))
            except ValueError: st.text("無效輸入，請輸入一個數字。")

    st.text("\n請輸入本次要投入/提領的金額，若需提領請輸入負值。")
    total_investment_base_currency = 0
    for currency in supported_currencies:
        amount = _get_numeric_input(f"{currency} 金額: ")
        if currency not in fx_rates:
            st.text(f"警告: 缺少 {currency}/{BASE_CURRENCY} 匯率，此筆金額將不被計入。")
            continue
        total_investment_base_currency += amount / fx_rates[currency]
        
    if not any(c in fx_rates for c in supported_currencies):
         st.text("警告：所有輸入貨幣的匯率均無法獲取，總投入/提領金額為0。")

    return total_investment_base_currency

def get_permissions(is_withdraw):
    """根據是提領還是投入，詢問對應的權限。"""
    if is_withdraw:
        st.text("\n偵測到提領操作。")
        buy = input("提領時，是否允許買入部分資產以達成平衡？ (y/n): ").lower().strip() in ['y', 'yes']
        return False, buy # sell=False, buy=True/False
    else:
        st.text("\n偵測到投入操作。")
        sell = input("投入時，是否允許賣出部分資產以達成平衡？ (y/n): ").lower().strip() in ['y', 'yes']
        return sell, False # sell=True/False, buy=False

def rebalance(investment_base, current_values_base, portfolio, is_withdraw, sell_allowed, buy_allowed):
    """通用再平衡計算函式。"""
    total_asset_base = current_values_base.sum() + investment_base
    target_values = portfolio * total_asset_base
    investment_diff = target_values - current_values_base

    if is_withdraw:
        if buy_allowed or not any(investment_diff > 0):
            return investment_diff
        else:
            sub_portfolio = portfolio[investment_diff < 0]
            sub_value = current_values_base[investment_diff < 0]
            sub_result = rebalance(investment_base, sub_value, sub_portfolio / sub_portfolio.sum(), is_withdraw, sell_allowed, buy_allowed)
            return sub_result.reindex(portfolio.index, fill_value=0)
    else: # is_invest
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
    sell_quantities = pd.Series(dtype=float)
    for asset, amount_base in sell_assets.items():
        currency = asset_currencies[asset]
        rate = fx_rates.get(currency, 1.0)
        price_local = prices[asset]
        # 賣出價值 (原始貨幣) = 賣出價值 (基準貨幣) * 匯率
        sell_value_local = abs(amount_base) * rate
        sell_quantities[asset] = sell_value_local / price_local if price_local > 0 else 0

    return buy_amounts_local, sell_quantities

# (繪圖函式 _draw_single_donut 和 plot_rebalancing_comparison_charts 維持不變，此處省略)
def plot_rebalancing_comparison_charts(before_ratios, after_ratios, target_ratios, filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10)); fig.set_facecolor('#2d2d3d')
    _draw_single_donut(ax1, before_ratios, target_ratios, "Before Rebalancing")
    _draw_single_donut(ax2, after_ratios, target_ratios, "After Rebalancing")
    plt.tight_layout(pad=3.0)
    plt.figure(fig, dpi = 300); return fig
    
def _draw_single_donut(ax, current_ratios, target_ratios, title):
    target_ratios = target_ratios.reindex(current_ratios.index); categories = current_ratios.index
    current_alloc_norm, target_alloc_norm = current_ratios.values, target_ratios.values
    colors = sns.color_palette('viridis', n_colors=len(categories)).as_hex(); bg_color = "#2d2d3dff"
    ax.set_facecolor(bg_color); ax.set_title(title, color='white', fontsize=20, pad=20); ax.set_aspect('equal')
    start_angle = 90; base_radius = 1.0; center_hole_radius = 0.6
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









# --- Streamlit 網頁應用主體 ---
def web_main():
    # 設定網頁標題和說明
    st.set_page_config(page_title="資產再平衡計算機", layout="wide")
    st.title("📈 資產組合再平衡計算機")
    st.markdown("""
    這個工具可以幫助您根據目標比例，計算出再平衡所需的交易。
    請上傳您的 Apple Numbers 追蹤檔案 (`.numbers`) 來開始。
    """)

    # 1. 檔案上傳元件
    uploaded_file = st.file_uploader("上傳您的 portfolio_tracker.numbers 檔案", type=["numbers"])

    if uploaded_file is not None:
        # 為了讓 numbers-parser 能讀取，需先將上傳的檔案暫存
        with open("temp_portfolio.numbers", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # 讀取檔案並進行初步驗證
        try:
            portfolio, quantities, tickers_list, table, df, doc = load_data_from_numbers("temp_portfolio.numbers")
            st.success("Numbers 檔案讀取成功！")
            st.dataframe(df) # 在網頁上顯示讀取到的表格
        except Exception as e:
            st.error(f"讀取 Numbers 檔案時出錯：{e}")
            st.stop() # 出錯則停止執行

        # 2. 互動式輸入元件
        st.header("設定再平衡參數")

        col1, col2 = st.columns(2)
        with col1:
            is_withdraw = st.radio("操作類型：", ('投入資金', '提領資金')) == '提領資金'

        with col2:
            if is_withdraw:
                buy_allowed = st.checkbox("提領時，允許買入部分資產以達成平衡？")
                sell_allowed = False
            else:
                sell_allowed = st.checkbox("投入時，允許賣出部分資產以達成平衡？")
                buy_allowed = False

        st.subheader("投入/提領金額")
        twd_invest = st.number_input("台幣 (TWD)", value=0.0, format="%.2f")
        usd_invest = st.number_input("美金 (USD)", value=0.0, format="%.2f")
        jpy_invest = st.number_input("日圓 (JPY)", value=0.0, format="%.2f")

        # 3. 執行按鈕
        if st.button("🚀 開始計算再平衡！", use_container_width=True):
            with st.spinner("正在獲取市場數據並執行計算..."):
                try:
                    # --- 執行您原有的核心邏輯 ---
                    prices, asset_currencies, fx_rates = get_asset_and_fx_data(tickers_list)

                    investment_base = (twd_invest / fx_rates.get('TWD', 1)) + \
                                      (usd_invest / fx_rates.get('USD', 1)) + \
                                      (jpy_invest / fx_rates.get('JPY', 1))

                    # ... (此處省略中間的計算過程，與您原本的 main 函式相同) ...

                    # 計算資產現值 (全部換算成基準貨幣 USD)
                    current_values_base = pd.Series(prices.values * quantities, index=prices.index)
                    # (後續所有計算...)
                    for asset, value in current_values_base.items():
                        currency = asset_currencies.get(asset, BASE_CURRENCY)
                        if currency != BASE_CURRENCY:
                            current_values_base[asset] /= fx_rates.get(currency, 1.0)
                    
                    # 5. 提領金額驗證
                    if is_withdraw:
                        total_withdrawal_base = abs(investment_base)
                        total_assets_base = current_values_base.sum()
                        if total_withdrawal_base > total_assets_base:
                            st.text(f"\n錯誤：欲提領金額 (約 ${total_withdrawal_base:,.2f}) 已超出資產總額 (約 ${total_assets_base:,.2f})。")
                            st.text("建議操作：請考慮賣出全部資產。"); exit()

                    # 6. 執行再平衡計算
                    st.text("\n正在計算再平衡計畫...")
                    result_base = rebalance(investment_base, current_values_base, portfolio, is_withdraw, sell_allowed, buy_allowed)
                    
                    # 7. 計算交易建議
                    buy_amounts_local, sell_quantities = calculate_transactions(result_base, prices, asset_currencies, fx_rates)

                    # --- 在網頁上顯示結果 ---
                    st.header("📊 計算結果")

                    # (顯示文字交易建議...)
                    st.subheader("--- 交易建議 ---")
                    for index in df['Ticker'].values:
                        column = np.where(df.columns == 'Shares to buy')[0][0]
                        row = np.where(df['Ticker'] == index)[0][0] + 1
                        table.write(row, column, 0, style = table.cell(row, column).style)    
                    if buy_amounts_local.empty and sell_quantities.empty:
                        st.text("無需進行任何交易。")
                    else:
                        # --- 修改開始：處理買入資產的顯示 ---
                        if not buy_amounts_local.empty:
                            # 1. 建立一個包含買入金額的 DataFrame
                            buy_df = pd.DataFrame(buy_amounts_local)
                            buy_df.columns = ['Amount_Local']

                            # 2. 計算建議購買的股數 (金額 / 價格)
                            #    使用 .reindex 確保價格與要買的資產對齊
                            aligned_prices = prices.reindex(buy_df.index)
                            buy_df['Shares_to_Buy'] = buy_df['Amount_Local'] / aligned_prices
                            
                            # 3. 建立用於顯示的格式化金額欄位
                            buy_df['Formatted_Amount'] = buy_df.apply(
                                lambda row: f"{asset_currencies[row.name]} {row['Amount_Local']:,.2f}",
                                axis=1
                            )
                            
                            # 4. 準備最終顯示的 DataFrame，選擇並重新命名欄位
                            display_df = buy_df[['Formatted_Amount', 'Shares_to_Buy']]
                            display_df.columns = ['買入金額', '建議股數']
                            
                            #5. 寫入 numbers 檔中 Shares to buy 欄
                            for index in display_df.index:
                                column = np.where(df.columns == 'Shares to buy')[0][0]
                                row = np.where(df['Ticker'] == index)[0][0] + 1
                                table.write(row, column, round(display_df['建議股數'][index],5), style = table.cell(row, column).style)
                            st.text("\n請買入：")
                            # .round(5) 讓股數的小數點後最多顯示5位
                            st.dataframe(display_df.round(5))
                        # --- 修改結束 ---
                        
                        if not sell_quantities.empty:
                            column = np.where(df.columns == 'Shares to buy')[0][0]
                            for index in sell_quantities.index:
                                row = np.where(df['Ticker'] == index)[0][0] + 1
                                table.write(row, column, round(-sell_quantities[index],5), style = table.cell(row, column).style)
                            st.text("\n請賣出 (股)：")
                            st.dataframe(sell_quantities.round(5))
                    doc.save("portfolio_tracker.numbers")
                    st.text("--- 交易建議結束 ---")
                    

                    # 繪製圖表並顯示
                    # (注意：plot函式需要修改，不再 plt.show()，而是回傳 figure 物件)
                    before_ratio = current_values_base / current_values_base.sum()
                    adjusted_values_base = current_values_base + result_base
                    adjusted_values_base[adjusted_values_base < 0] = 0
                    after_ratio = adjusted_values_base / adjusted_values_base.sum() if adjusted_values_base.sum() > 0 else before_ratio
                    
                    fig = plot_rebalancing_comparison_charts(
                        before_ratios=before_ratio,
                        after_ratios=after_ratio,
                        target_ratios=portfolio,
                        filename="rebalancing_side_by_side.png"
                    )
                    st.pyplot(fig)

                    st.success("計算完成！")

                except Exception as e:
                    st.error(f"計算過程中發生錯誤：{e}")



if __name__ == '__main__':
    web_main()
