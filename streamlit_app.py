import streamlit as st
import time
import datetime
# fx_ai_trader_core.py から関数と定数をインポート
from fx_ai_trader_core import (
    get_ohlcv_data,
    calculate_indicators,
    generate_signals,
    plot_chart_with_signals,
    TICKER,
    INTERVAL,
    PERIOD,
    ALLIGATOR_JAW_LEN,
    ALLIGATOR_JAW_SHIFT,
    ALLIGATOR_LIPS_LEN,
    ALLIGATOR_TEETH_LEN,
    ALLIGATOR_TEETH_SHIFT,
    ALLIGATOR_LIPS_SHIFT,
    ADX_LENGTH,
    ADX_TREND_THRESHOLD,
    ADX_RANGE_THRESHOLD,
    MIN_PIPS_CHANGE_FOR_SIGNAL,
    SIGNAL_CHECK_TIMEFRAME_MINS,
    STOP_LOSS_PIPS,
    TAKE_PROFIT_PIPS
)

st.set_page_config(layout="wide") # ページレイアウトをワイドに設定
st.title("FX AI Trader (リアルタイムデモ)")

# INTERVAL定数から時間足（分）を計算
if INTERVAL.endswith('m'):
    INTERVAL_MINUTES = int(INTERVAL[:-1])
elif INTERVAL.endswith('h'):
    INTERVAL_MINUTES = int(INTERVAL[:-1]) * 60
elif INTERVAL.endswith('d'):
    INTERVAL_MINUTES = int(INTERVAL[:-1]) * 24 * 60
else:
    INTERVAL_MINUTES = 1 # デフォルト値

# リアルタイム更新のプレースホルダー
placeholder = st.empty()

# リアルタイム更新ループ
while True:
    with placeholder.container():
        st.write(f"最終更新: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # データ取得
        data_df = get_ohlcv_data(TICKER, INTERVAL, PERIOD)

        if data_df.empty:
            st.warning("データ取得に失敗しました。次の更新を待っています...")
            time.sleep(60) # 失敗時も待機して再試行
            continue

        # インジケーター計算
        data_df = calculate_indicators(data_df)

        # シグナル生成
        data_df = generate_signals(data_df, INTERVAL_MINUTES)
        
        # チャート表示
        fig = plot_chart_with_signals(data_df, INTERVAL_MINUTES)
        st.pyplot(fig) # StreamlitでmatplotlibのFigureを表示

        # 最新のデータフレームの一部を表示（オプション）
        st.subheader("最新データ（一部）")
        st.dataframe(data_df.tail())

        # 次の更新まで待機
        st.write(f"✅ 次回更新まで {INTERVAL_MINUTES} 分待機します...")
        time.sleep(INTERVAL_MINUTES * 1) # データ間隔に合わせて更新
