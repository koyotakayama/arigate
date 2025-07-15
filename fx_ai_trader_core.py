import pandas as pd
import numpy as np
import yfinance as yf
import mplfinance as mpf
import ta  # Technical Analysis Library (pip install ta)

# --- 1. 定数と初期設定 ---
TICKER = 'USDJPY=X'  # 通貨ペア (Yahoo Financeの場合、FXは'=X'を付ける)
INTERVAL = '5m'     # データの間隔 ('1m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
PERIOD = '1d'        # データの期間 (e.g., '1d', '5d', '60d', '1y', '5y', 'max')
# 注意: 1mは7日間、5m/15mは60日間、60m/1hは730日間までしか取得できない制約がある.

# アリゲーターインジケーターの定数 (Williams Alligator)
ALLIGATOR_JAW_LEN = 13   # 顎 (Jaw) の期間
ALLIGATOR_TEETH_LEN = 8  # 歯 (Teeth) の期間
ALLIGATOR_LIPS_LEN = 5   # 唇 (Lips) の期間
ALLIGATOR_JAW_SHIFT = 8  # 顎 (Jaw) のシフト
ALLIGATOR_TEETH_SHIFT = 5 # 歯 (Teeth) のシフト
ALLIGATOR_LIPS_SHIFT = 3 # 唇 (Lips) のシフト

# ADX (Average Directional Index) の定数
ADX_LENGTH = 14          # ADXの計算期間
ADX_TREND_THRESHOLD = 25 # トレンド判定のADX閾値
ADX_RANGE_THRESHOLD = 20 # レンジ判定のADX閾値

# シグナル生成のためのその他の定数
MIN_PIPS_CHANGE_FOR_SIGNAL = 5 # エントリーを検討する最小価格変動 (pips)
SIGNAL_CHECK_TIMEFRAME_MINS = 15 # 何分間の値動きをチェックするか (データ間隔と合わせるか、倍数にする)
STOP_LOSS_PIPS = 30 # デフォルトの損切り幅 (pips)
TAKE_PROFIT_PIPS = 60 # デフォルトの利確幅 (pips)


# --- 2. データ取得モジュール ---
# Yahoo FinanceからOHLCVデータを取得する関数 (これはコメントです)
def get_ohlcv_data(ticker, interval, period):
    """
    Yahoo FinanceからOHLCVデータを取得する関数
    :param ticker: 通貨ペアのティッカーシンボル (例: 'EURUSD=X')
    :param interval: データの間隔 (例: '1m', '5m', '15m')
    :param period: データの期間 (例: '7d', '60d')
    :return: pandas DataFrame (OHLCV, インデックスはDatetimeIndex)
    """
    # Streamlitで実行されることを想定し、冗長なprintはコメントアウト
    # print(f"データ取得中: {ticker}, 間隔: {interval}, 期間: {period}...")
    try:
        data = yf.download(ticker, interval=interval, period=period, auto_adjust=True)
        
        if data.empty:
            # print(f"エラー: {ticker} のデータが見つかりません.期間と間隔を確認してください.")
            return pd.DataFrame()

        # --- デバッグ用: 取得直後のDataFrameのインデックスとカラムを確認 ---
        # print(f"デバッグ: 取得直後のDataFrameインデックス: {data.index.name}, タイプ: {type(data.index)}")
        # print(f"デバッグ: 取得直後のDataFrameカラム: {data.columns.tolist()}")
        # --- デバッグ用ここまで ---

        # インデックスを一旦カラムに変換する
        data_reset = data.reset_index()

        # --- デバッグ用: reset_index後のDataFrameカラムを確認 ---
        # print(f"デバッグ: reset_index後のDataFrameカラム: {data_reset.columns.tolist()}")
        # --- デバッグ用ここまで ---

        datetime_column_name = None
        column_candidates_base = ['Datetime', 'Date', 'index', 'datetime'] 

        for col in data_reset.columns:
            col_name_base = None
            if isinstance(col, tuple): # MultiIndexの場合
                if col and isinstance(col[0], str):
                    col_name_base = col[0]
            elif isinstance(col, str): # 通常の文字列カラム名の場合
                col_name_base = col
            
            if col_name_base in column_candidates_base:
                try:
                    temp_series = pd.to_datetime(data_reset[col], errors='coerce')
                    if not temp_series.isnull().all() and not temp_series.empty:
                        data_reset[col] = temp_series # 変換結果を元のカラムに適用
                        datetime_column_name = col # 元のカラム名 (タプルまたは文字列) を保持
                        break
                except Exception as ex:
                    # print(f"デバッグ: カラム '{col}' ({col_name_base}) の日時変換失敗: {ex}")
                    continue
        
        if datetime_column_name is None:
            # print("致命的なエラー: 日時情報を持つ有効なカラムが見つかりませんでした.データ処理を中断します.")
            return pd.DataFrame()

        # 日時情報が無効な行を削除する
        data_reset.dropna(subset=[datetime_column_name], inplace=True)
        
        # 新しい DatetimeIndex を設定し、そのインデックス名を明示的に None に設定する
        data = data_reset.set_index(datetime_column_name)
        data.index.name = None # ここが非常に重要!インデックス名を強制的にNoneにする
        
        # --- カラム名を安全に正規化 ---
        required_cols_lower = ['open', 'high', 'low', 'close', 'volume']
        
        rename_map = {}
        for original_col in data.columns:
            col_name_base = None
            if isinstance(original_col, tuple):
                if original_col and isinstance(original_col[0], str):
                    col_name_base = original_col[0].lower()
            elif isinstance(original_col, str):
                col_name_base = original_col.lower()

            if col_name_base in required_cols_lower:
                new_name = ''
                if col_name_base == 'open': new_name = 'Open'
                elif col_name_base == 'high': new_name = 'High'
                elif col_name_base == 'low': new_name = 'Low'
                elif col_name_base == 'close': new_name = 'Close'
                elif col_name_base == 'volume': new_name = 'Volume'
                rename_map[original_col] = new_name
            else:
                # print(f"警告: 認識できないカラムが見つかりました: {original_col}. スキップします.")
                pass # Skip unrecognized columns
            
        data = data.rename(columns=rename_map)

        final_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        data = data[[col for col in final_cols if col in data.columns]]
        
        if not all(col in data.columns for col in final_cols):
            # print("エラー: 最終的なOHLCVカラムが揃っていません.データ処理を中断します.")
            return pd.DataFrame()

        # print("データ取得完了.")
        return data
    except Exception as e:
        # print(f"データ取得エラー: {e}")
        return pd.DataFrame()

# --- 3. インジケーター計算モジュールに必要なヘルパー関数 ---

# SMMA (Smoothed Moving Average) の計算関数
def _calculate_smma(series, length):
    """
    Smoothed Moving Average (SMMA) を計算するヘルパー関数.
    これはRSIの計算などで使われるEWMAに似ていますが, 計算式が少し異なります.
    """
    smma = pd.Series(np.nan, index=series.index)
    
    if len(series) >= length and not series.iloc[:length].isnull().any():
        smma.iloc[length - 1] = series.iloc[:length].mean()
    else:
        return smma

    for i in range(length, len(series)):
        smma.iloc[i] = (smma.iloc[i-1] * (length - 1) + series.iloc[i]) / length
    return smma

# ウィリアムズフラクタル (Williams Fractals) の計算関数
def _calculate_fractal(df_input):
    """
    ウィリアムズフラクタル (Williams Fractals) を計算するヘルパー関数.
    上向きフラクタルと下向きフラクタルを識別します.
    """
    df = df_input.copy()
    
    df['FR_U'] = np.nan
    df['FR_L'] = np.nan

    highs = df['High']
    lows = df['Low']

    is_upper_fractal = (
        (highs.shift(2) > highs.shift(4)) &
        (highs.shift(2) > highs.shift(3)) &
        (highs.shift(2) > highs.shift(1)) &
        (highs.shift(2) > highs)
    )
    df.loc[is_upper_fractal.shift(-2).fillna(False), 'FR_U'] = highs.shift(2)[is_upper_fractal.shift(-2).fillna(False)]

    is_lower_fractal = (
        (lows.shift(2) < lows.shift(4)) &
        (lows.shift(2) < lows.shift(3)) &
        (lows.shift(2) < lows.shift(1)) &
        (lows.shift(2) < lows)
    )
    df.loc[is_lower_fractal.shift(-2).fillna(False), 'FR_L'] = lows.shift(2)[is_lower_fractal.shift(-2).fillna(False)]
            
    return df[['FR_U', 'FR_L']]

# メインのインジケーター計算関数
def calculate_indicators(df_input):
    # print("インジケーター計算中.")
    df = df_input.copy()

    # --- OHLCVカラム名の正規化 ---
    new_columns = []
    for col in df.columns:
        if isinstance(col, tuple):
            if col and isinstance(col[0], str):
                new_columns.append(col[0])
            else:
                new_columns.append(str(col))
        else:
            new_columns.append(col)
    df.columns = new_columns
    
    df.rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
    }, inplace=True)

    required_ohlcv = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_ohlcv):
        # print(f"エラー: 必要なOHLCVカラムが揃っていません.現在のカラム: {df.columns.tolist()}")
        return pd.DataFrame()
    # --- OHLCVカラム名の正規化ここまで ---

    # Ensure 'Close', 'High', 'Low' columns are 1-dimensional Series
    # This is the core of the fix for "Data must be 1-dimensional"
    close_series = df['Close']
    if isinstance(close_series, pd.DataFrame):
        close_series = close_series.squeeze() # Convert single-column DataFrame to Series
    if close_series.ndim > 1:
        close_series = np.squeeze(close_series) # Fallback for unexpected 2D numpy array-like data
    
    high_series = df['High']
    if isinstance(high_series, pd.DataFrame):
        high_series = high_series.squeeze()
    if high_series.ndim > 1:
        high_series = np.squeeze(high_series)

    low_series = df['Low']
    if isinstance(low_series, pd.DataFrame):
        low_series = low_series.squeeze()
    if low_series.ndim > 1:
        low_series = np.squeeze(low_series)


    # 移動平均の計算
    df['SMA_20'] = ta.trend.sma_indicator(close=close_series, window=20)
    df['EMA_10'] = ta.trend.ema_indicator(close=close_series, window=10)
    df['EMA_50'] = ta.trend.ema_indicator(close=close_series, window=50)
    df['EMA_100'] = ta.trend.ema_indicator(close=close_series, window=100)
    df['EMA_200'] = ta.trend.ema_indicator(close=close_series, window=200)

    # RSI (Relative Strength Index)
    df['RSI'] = ta.momentum.rsi(close=close_series, window=14)

    # MACD (Moving Average Convergence Divergence)
    df['MACD'] = ta.trend.macd(close=close_series)
    df['MACD_Signal'] = ta.trend.macd_signal(close=close_series)
    df['MACD_Diff'] = ta.trend.macd_diff(close=close_series)

    # ストキャスティクス (Stochastic Oscillator)
    df['STOCH_K'] = ta.momentum.stoch(high=high_series, low=low_series, close=close_series, window=14)
    df['STOCH_D'] = ta.momentum.stoch_signal(high=high_series, low=low_series, close=close_series, window=14)

    # ボリンジャーバンド (Bollinger Bands)
    df['BB_Upper'] = ta.volatility.bollinger_hband(close=close_series, window=20)
    df['BB_Lower'] = ta.volatility.bollinger_lband(close=close_series, window=20)
    df['BB_Middle'] = ta.volatility.bollinger_mavg(close=close_series, window=20)

    # フィボナッチリトレースメントは動的なため、ここでは計算せずシグナル生成時に動的に扱うか、
    # 描画時に別途計算する.

    # アリゲーターインジケーターの計算
    # 中央値 (Median Price) = (High + Low) / 2
    median_price = (high_series + low_series) / 2 

    df['Alligator_Jaw'] = _calculate_smma(median_price, ALLIGATOR_JAW_LEN).shift(ALLIGATOR_JAW_SHIFT)
    df['Alligator_Teeth'] = _calculate_smma(median_price, ALLIGATOR_TEETH_LEN).shift(ALLIGATOR_TEETH_SHIFT)
    df['Alligator_Lips'] = _calculate_smma(median_price, ALLIGATOR_LIPS_LEN).shift(ALLIGATOR_LIPS_SHIFT)

    # フラクタル (Fractals) の計算
    fractal_results = _calculate_fractal(df)
    df['FR_U'] = fractal_results['FR_U']
    df['FR_L'] = fractal_results['FR_L']

    # ADX (Average Directional Index) の計算
    df[f'ADX_{ADX_LENGTH}'] = ta.trend.adx(high=high_series, low=low_series, close=close_series, window=ADX_LENGTH)

    # print("インジケーター計算完了.")
    return df

# --- 4. シグナル生成モジュール ---
def generate_signals(df_input, interval_minutes):
    """
    定義されたロジックに基づいてエントリーシグナルを生成する関数
    :param df_input: OHLCVデータとインジケーターを含むDataFrame
    :param interval_minutes: データの時間足の間隔 (分)
    :return: シグナル情報が追加されたDataFrame
    """
    df = df_input.copy()
    df['Signal'] = 'None'
    df['Entry_Price'] = np.nan
    df['StopLoss_Price'] = np.nan
    df['TakeProfit_Price'] = np.nan
    # print("シグナル生成中.")

    # アリゲーターの各ラインを短くするために変数に格納 (calculate_indicatorsで設定したカラム名を使用)
    jaw = df['Alligator_Jaw']
    teeth = df['Alligator_Teeth']
    lips = df['Alligator_Lips']
    
    adx = df[f'ADX_{ADX_LENGTH}']
    upper_fractal = df['FR_U']
    lower_fractal = df['FR_L']

    # ピップス計算のための乗数 (USD/JPYなら0.01、EUR/USDなら0.0001)
    # ここではEURUSD=Xを想定して0.0001を使用.通貨ペアによって調整が必要です.
    # 例: JPYを含むペアは0.01、それ以外は0.0001が一般的
    pip_multiplier = 0.0001 if 'JPY' not in TICKER else 0.01

    # 過去の値動きをチェックするローソク足の本数
    num_bars_to_check = max(1, int(SIGNAL_CHECK_TIMEFRAME_MINS / interval_minutes))


    for i in range(len(df)):
        # インジケーター計算に必要なデータが揃っていない場合はスキップ
        if i < max(ALLIGATOR_JAW_LEN + ALLIGATOR_JAW_SHIFT, ADX_LENGTH, num_bars_to_check + 2): # フラクタルは前後2本必要なので+2
            continue

        current_adx = adx.iloc[i]
        current_close = df['Close'].iloc[i]

        # 過去N本のローソク足の高値と安値の差を計算 (pips単位)
        past_period_high = df['High'].iloc[i - num_bars_to_check : i + 1].max()
        past_period_low = df['Low'].iloc[i - num_bars_to_check : i + 1].min()
        price_change_pips = (past_period_high - past_period_low) / pip_multiplier

        # アリゲーターの状態判断
        # NaNチェックを追加
        if pd.isna(lips.iloc[i]) or pd.isna(teeth.iloc[i]) or pd.isna(jaw.iloc[i]):
            continue # いずれかのラインがNaNならスキップ

        alligator_trend_up = (lips.iloc[i] > teeth.iloc[i] > jaw.iloc[i])
        alligator_trend_down = (jaw.iloc[i] > teeth.iloc[i] > lips.iloc[i])
        is_sleeping_threshold = 0.5 * pip_multiplier # 例: 0.5pips以内
        alligator_sleeping = (abs(lips.iloc[i] - teeth.iloc[i]) < is_sleeping_threshold and
                              abs(teeth.iloc[i] - jaw.iloc[i]) < is_sleeping_threshold)

        # フラクタルによるブレイクアウト/反転の判断 (簡略化)
        is_upper_fractal_present = pd.notna(upper_fractal.iloc[i-2])
        is_lower_fractal_present = pd.notna(lower_fractal.iloc[i-2])
        
        # エントリーポイントの決定と利確・損切りの設定
        # --- シグナルを出す共通条件: 最小ピップス変化量 ---
        if price_change_pips >= MIN_PIPS_CHANGE_FOR_SIGNAL:
            # --- トレンド相場でのシグナル ---
            if current_adx >= ADX_TREND_THRESHOLD:
                # 上昇トレンド順張り
                if alligator_trend_up and is_upper_fractal_present and \
                   current_close > upper_fractal.iloc[i-2]:
                    df.loc[df.index[i], 'Signal'] = 'Buy_Trend'
                    df.loc[df.index[i], 'Entry_Price'] = current_close
                    
                    # 損切り価格の計算: フラクタル安値またはアリゲーターJaw、いずれか低い方
                    sl_candidate_fractal = lower_fractal.iloc[i-2] if is_lower_fractal_present else current_close - (STOP_LOSS_PIPS * pip_multiplier)
                    df.loc[df.index[i], 'StopLoss_Price'] = min(sl_candidate_fractal, jaw.iloc[i])
                    
                    df.loc[df.index[i], 'TakeProfit_Price'] = current_close + (TAKE_PROFIT_PIPS * pip_multiplier)

                # 下降トレンド順張り
                elif alligator_trend_down and is_lower_fractal_present and \
                     current_close < lower_fractal.iloc[i-2]:
                    df.loc[df.index[i], 'Signal'] = 'Sell_Trend'
                    df.loc[df.index[i], 'Entry_Price'] = current_close
                    
                    # 損切り価格の計算: フラクタル高値またはアリゲーターJaw、いずれか高い方
                    sl_candidate_fractal = upper_fractal.iloc[i-2] if is_upper_fractal_present else current_close + (STOP_LOSS_PIPS * pip_multiplier)
                    df.loc[df.index[i], 'StopLoss_Price'] = max(sl_candidate_fractal, jaw.iloc[i])

                    df.loc[df.index[i], 'TakeProfit_Price'] = current_close - (TAKE_PROFIT_PIPS * pip_multiplier)

            # --- レンジ相場でのシグナル ---
            elif current_adx < ADX_RANGE_THRESHOLD:
                if alligator_sleeping: # アリゲーターが寝ている状態
                    # レンジ上限からの反転下降
                    if is_upper_fractal_present and current_close < upper_fractal.iloc[i-2] and \
                       df['Open'].iloc[i] >= upper_fractal.iloc[i-2]:
                        df.loc[df.index[i], 'Signal'] = 'Sell_Range'
                        df.loc[df.index[i], 'Entry_Price'] = current_close
                        df.loc[df.index[i], 'StopLoss_Price'] = upper_fractal.iloc[i-2] + (5 * pip_multiplier)
                        df.loc[df.index[i], 'TakeProfit_Price'] = current_close - (TAKE_PROFIT_PIPS * pip_multiplier)

                    # レンジ下限からの反転上昇
                    elif is_lower_fractal_present and current_close > lower_fractal.iloc[i-2] and \
                         df['Open'].iloc[i] <= lower_fractal.iloc[i-2]:
                        df.loc[df.index[i], 'Signal'] = 'Buy_Range'
                        df.loc[df.index[i], 'Entry_Price'] = current_close
                        df.loc[df.index[i], 'StopLoss_Price'] = lower_fractal.iloc[i-2] - (5 * pip_multiplier)
                        df.loc[df.index[i], 'TakeProfit_Price'] = current_close + (TAKE_PROFIT_PIPS * pip_multiplier)
    # print("シグナル生成完了.")
    return df

# --- 5. チャート表示モジュール ---
def plot_chart_with_signals(df, interval_minutes, title="FX Chart with AI Signals"):
    """
    ローソク足チャートにインジケーターとシグナルを表示する関数
    :param df: OHLCVデータとインジケーターを含むDataFrame
    :param interval_minutes: データの時間足の間隔 (分)
    :param title: チャートのタイトル
    :return: matplotlib.figure.Figure オブジェクト
    """
    # print("チャート描画中.")

    # アリゲーターのライン名
    jaw_label = 'Alligator_Jaw'
    teeth_label = 'Alligator_Teeth'
    lips_label = 'Alligator_Lips'

    # プロットに追加するインジケーター設定
    apds = [
        # アリゲーター Jaw (青)
        mpf.make_addplot(df[jaw_label], panel=0, type='line',
                         color='blue', ylabel='Alligator', width=0.8),
        # アリゲーター Teeth (赤)
        mpf.make_addplot(df[teeth_label], panel=0, type='line',
                         color='red', width=0.8),
        # アリゲーター Lips (緑)
        mpf.make_addplot(df[lips_label], panel=0, type='line',
                         color='green', width=0.8),

        # ADXとその閾値ライン
        mpf.make_addplot(df[f'ADX_{ADX_LENGTH}'], panel=1, color='purple', ylabel='ADX'),
        mpf.make_addplot(pd.Series(ADX_TREND_THRESHOLD, index=df.index), panel=1, type='line', color='darkorange', width=0.7, linestyle='--', label='ADX Trend Threshold'),
        mpf.make_addplot(pd.Series(ADX_RANGE_THRESHOLD, index=df.index), panel=1, type='line', color='cyan', width=0.7, linestyle='--', label='ADX Range Threshold'),
    ]

    # フラクタルのプロット
    if not df['FR_U'].dropna().empty:
        apds.append(mpf.make_addplot(df['FR_U'], type='scatter', marker='^', markersize=100, color='brown', panel=0, alpha=0.7, label='Upper Fractal'))
    if not df['FR_L'].dropna().empty:
        apds.append(mpf.make_addplot(df['FR_L'], type='scatter', marker='v', markersize=100, color='darkgreen', panel=0, alpha=0.7, label='Lower Fractal'))


    # シグナルのプロット
    buy_entry_plot_series = pd.Series(np.nan, index=df.index)
    sell_entry_plot_series = pd.Series(np.nan, index=df.index)

    buy_signal_indices = df[df['Signal'].str.contains('Buy') & pd.notna(df['Entry_Price'])].index
    buy_entry_plot_series.loc[buy_signal_indices] = df.loc[buy_signal_indices, 'Entry_Price']
    
    sell_signal_indices = df[df['Signal'].str.contains('Sell') & pd.notna(df['Entry_Price'])].index
    sell_entry_plot_series.loc[sell_signal_indices] = df.loc[sell_signal_indices, 'Entry_Price']

    if not buy_entry_plot_series.dropna().empty:
        apds.append(mpf.make_addplot(buy_entry_plot_series, panel=0, type='scatter', marker='^', color='black', markersize=200, label='Buy Signal'))
    if not sell_entry_plot_series.dropna().empty:
        apds.append(mpf.make_addplot(sell_entry_plot_series, panel=0, type='scatter', marker='v', color='blue', markersize=200, label='Sell Signal'))

    # 利確・損切りラインのプロット
    tp_plot_series = pd.Series(np.nan, index=df.index)
    sl_plot_series = pd.Series(np.nan, index=df.index)

    signal_indices = df[df['Signal'] != 'None'].index
    for idx in signal_indices:
        entry_price = df.loc[idx, 'Entry_Price']
        sl_price = df.loc[idx, 'StopLoss_Price']
        tp_price = df.loc[idx, 'TakeProfit_Price']
        
        if pd.notna(entry_price):
            if pd.notna(tp_price):
                tp_plot_series.loc[idx] = tp_price
            if pd.notna(sl_price):
                sl_plot_series.loc[idx] = sl_price
    
    if not tp_plot_series.dropna().empty:
        apds.append(mpf.make_addplot(tp_plot_series, panel=0, type='scatter', marker='_', color='lime', markersize=100, label='Take Profit'))
    if not sl_plot_series.dropna().empty:
        apds.append(mpf.make_addplot(sl_plot_series, panel=0, type='scatter', marker='_', color='orangered', markersize=100, label='Stop Loss'))
        
    # mpf.plotでの描画
    fig, axes = mpf.plot(df,
                         type='candle',
                         style='yahoo',
                         title=title,
                         ylabel='Price',
                         volume=True,
                         ylabel_lower='Volume',
                         addplot=apds,
                         returnfig=True,
                         figscale=1.5,
                         tight_layout=True
                        )
    
    # mpf.show() はStreamlitでは使わず、figを返す
    return fig
