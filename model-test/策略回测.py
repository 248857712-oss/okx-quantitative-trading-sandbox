import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import json
import sys
import os
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# 加入环境变量，复用main.py的配置和行情逻辑
sys.path.append(os.getcwd())
from main import load_config, create_proxy_session
from ccxt import okx

# ===================== 1. 核心配置（和实战策略完全一致） =====================
CONFIG = load_config()
SYMBOL = CONFIG["okx"]["symbol"]
TIMEFRAME = "1h"  # 回测周期，和实战一致
DATA_LIMIT = 4320  # 回测数据量：1h×4320=6个月，足够验证策略
LEVERAGE = 10  # 和实战一致，不改动
FEE_RATE = 0.0005  # OKX合约手续费率（模拟盘/实盘通用）


# ===================== 2. 历史数据获取（批量拉取，支持本地缓存） =====================
def get_historical_data(symbol=SYMBOL, tf=TIMEFRAME, limit=DATA_LIMIT):
    """批量拉取历史K线，支持本地缓存，避免重复调用API"""
    cache_file = f"historical_data_{tf}_{limit}.csv"
    # 优先读取本地缓存，提速
    if os.path.exists(cache_file):
        print(f"📊 读取本地缓存数据：{cache_file}")
        df = pd.read_csv(cache_file, index_col=0)
        df["ts"] = pd.to_datetime(df["ts"])
        return df.dropna()

    # 无缓存则从OKX拉取
    print(f"📡 从OKX拉取{limit}根{tf}K线数据（约{limit // 24 / 30:.1f}个月）")
    PROXY_SETTINGS = CONFIG["proxy"]
    okx_client = okx({
        "apiKey": CONFIG["okx"]["api_key"],
        "secret": CONFIG["okx"]["api_secret"],
        "password": CONFIG["okx"]["api_passphrase"],
        "enableRateLimit": True,
        "sandbox": CONFIG["okx"]["is_sim"],
        "proxies": PROXY_SETTINGS,
        "options": {"defaultType": "swap"}
    })
    okx_client.load_markets()

    # 时间周期小写，适配OKX接口
    timeframe_mapping = {'1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m', '1h': '1h', '1d': '1d'}
    okx_tf = timeframe_mapping.get(tf, '1h')

    # 批量拉取（ccxt单次最多1000，循环拉取）
    all_ohlcv = []
    since = None
    batch_size = 1000
    while len(all_ohlcv) < limit:
        ohlcv = okx_client.fetch_ohlcv(symbol, okx_tf, since, min(batch_size, limit - len(all_ohlcv)))
        if not ohlcv:
            break
        all_ohlcv.extend(ohlcv)
        since = ohlcv[-1][0] + 1  # 下一批起始时间

    # 整理数据
    df = pd.DataFrame(all_ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
    for col in ['open', 'high', 'low', 'close', 'vol']:
        df[col] = pd.to_numeric(df[col])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df = df.drop_duplicates(subset=['ts']).reset_index(drop=True)

    # 保存本地缓存
    df.to_csv(cache_file, encoding='utf-8')
    print(f"✅ 数据拉取完成，共{len(df)}条，已缓存到本地")
    return df


# ===================== 3. 策略核心逻辑（和main.py完全一致，保证回测真实） =====================
def strategy_logic(df, params):
    """
    复用原策略核心逻辑：特征生成+模型训练+信号输出+交易执行
    params：调优后的最优参数
    """
    df_copy = df.copy()
    # 1. 特征工程（和原策略一致）
    df_copy['ret'] = (df_copy['close'] - df_copy['open']) / df_copy['open']
    df_copy['range'] = (df_copy['high'] - df_copy['low']) / df_copy['open']
    df_copy['vol_ratio'] = df_copy['vol'] / df_copy['vol'].rolling(20).mean()
    # 布林带计算（和原策略一致）
    df_copy['boll_mid'] = df_copy['close'].rolling(window=params['boll_window']).mean()
    df_copy['boll_std'] = df_copy['close'].rolling(window=params['boll_window']).std()
    df_copy['boll_upper'] = df_copy['boll_mid'] + params['boll_dev'] * df_copy['boll_std']
    df_copy['boll_lower'] = df_copy['boll_mid'] - params['boll_dev'] * df_copy['boll_std']
    # 标签：下一根K线收盘价上涨为1，下跌为0
    df_copy['y'] = (df_copy['close'].shift(-1) > df_copy['close']).astype(int)
    df_copy = df_copy.dropna().reset_index(drop=True)

    if len(df_copy) < 200:
        return pd.DataFrame(), 0  # 样本不足

    # 2. 划分训练集和测试集（前70%训练，后30%回测，避免未来函数）
    train_size = int(len(df_copy) * 0.7)
    train_df = df_copy.iloc[:train_size]
    test_df = df_copy.iloc[train_size:].reset_index(drop=True)

    # 3. 模型训练（和原策略一致：LR+RF加权）
    X_train = train_df[['ret', 'range', 'vol_ratio']]
    y_train = train_df['y']
    X_test = test_df[['ret', 'range', 'vol_ratio']]

    lr = LogisticRegression(random_state=42, max_iter=200)
    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    # 4. 模拟交易（完全对齐原策略开仓平仓逻辑）
    trades = []
    position = 0  # 0无仓，1多仓
    entry_price = 0.0
    boll_lower = 0.0

    for idx, row in test_df.iterrows():
        # 更新布林带下轨（极端行情平仓用）
        boll_lower = row['boll_lower']
        current_price = row['close']

        # 生成交易信号
        x = pd.DataFrame([[row['ret'], row['range'], row['vol_ratio']]], columns=['ret', 'range', 'vol_ratio'])
        lr_prob = lr.predict_proba(x)[0][1]
        rf_prob = rf.predict_proba(x)[0][1]
        weighted_prob = lr_prob * params['lr_weight'] + rf_prob * params['rf_weight']

        # ===== 开仓逻辑（和原策略一致） =====
        if position == 0:
            # 多单开仓：加权概率≥开仓阈值
            if weighted_prob >= params['vote_threshold']:
                position = 1
                entry_price = current_price
                trade = {
                    'time': row['ts'],
                    'type': 'open_long',
                    'entry_price': entry_price,
                    'exit_price': None,
                    'profit_ratio': 0,
                    'status': 'holding'
                }
                trades.append(trade)

        # ===== 平仓逻辑（和原策略完全一致：止盈+止损+布林带下轨强制平仓） =====
        elif position == 1:
            profit_ratio = (current_price - entry_price) / entry_price
            # 1. 布林带下轨强制平仓（最高优先级）
            if boll_lower > 0 and current_price < boll_lower:
                trades[-1]['exit_price'] = current_price
                trades[-1]['profit_ratio'] = profit_ratio - 2 * FEE_RATE  # 扣除开平仓手续费
                trades[-1]['status'] = 'closed_force'
                position = 0
            # 2. 止盈平仓（双盈利阈值达标）
            elif profit_ratio >= params['min_profit_threshold'] and profit_ratio >= params['target_profit_ratio']:
                trades[-1]['exit_price'] = current_price
                trades[-1]['profit_ratio'] = profit_ratio - 2 * FEE_RATE
                trades[-1]['status'] = 'closed_tp'
                position = 0
            # 3. 止损平仓（亏损阈值达标）
            elif profit_ratio <= -params['min_loss_threshold']:
                trades[-1]['exit_price'] = current_price
                trades[-1]['profit_ratio'] = profit_ratio - 2 * FEE_RATE
                trades[-1]['status'] = 'closed_sl'
                position = 0

    # 整理交易记录
    trades_df = pd.DataFrame(trades)
    return trades_df, train_size


# ===================== 4. 核心回测指标计算（6大关键指标+详细统计） =====================
def calculate_backtest_metrics(trades_df, test_df):
    """计算量化必看的核心回测指标，输出详细报告"""
    if trades_df.empty or len(trades_df) == 0:
        print("❌ 无有效交易记录，无法计算指标")
        return {}

    # 提取有效平仓交易
    closed_trades = trades_df[trades_df['status'].str.contains('closed')].copy()
    if len(closed_trades) == 0:
        print("❌ 无平仓交易记录")
        return {}

    # 基础收益统计
    closed_trades['profit'] = closed_trades['profit_ratio'] * LEVERAGE  # 杠杆放大收益
    total_trades = len(closed_trades)
    win_trades = closed_trades[closed_trades['profit'] > 0]
    lose_trades = closed_trades[closed_trades['profit'] < 0]
    win_rate = len(win_trades) / total_trades

    # 核心指标计算
    total_profit = closed_trades['profit'].sum()
    avg_win = win_trades['profit'].mean() if len(win_trades) > 0 else 0
    avg_lose = lose_trades['profit'].mean() if len(lose_trades) > 0 else 0
    profit_loss_ratio = abs(avg_win / avg_lose) if avg_lose != 0 else 0

    # 年化收益率（按回测周期计算）
    test_start = test_df['ts'].iloc[0]
    test_end = test_df['ts'].iloc[-1]
    test_days = (test_end - test_start).days
    annual_return = (1 + total_profit) ** (365 / test_days) - 1 if test_days > 0 else 0

    # 最大回撤（按累计收益计算）
    closed_trades['cum_profit'] = closed_trades['profit'].cumsum()
    peak = closed_trades['cum_profit'].expanding(min_periods=1).max()
    drawdown = (closed_trades['cum_profit'] - peak) / peak
    max_drawdown = drawdown.min()

    # 夏普比率（假设无风险利率3%）
    risk_free_rate = 0.03
    profit_std = closed_trades['profit'].std()
    sharpe_ratio = (annual_return - risk_free_rate) / (profit_std * np.sqrt(365 / test_days)) if profit_std != 0 else 0

    # 卡玛比率（年化收益/最大回撤绝对值）
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

    # 手续费占比
    total_fee = total_trades * 2 * FEE_RATE * LEVERAGE  # 开仓+平仓手续费
    fee_ratio = total_fee / (total_profit + total_fee) if (total_profit + total_fee) != 0 else 0

    # 整理指标
    metrics = {
        "回测周期": f"{test_start.strftime('%Y-%m-%d')} ~ {test_end.strftime('%Y-%m-%d')}",
        "回测天数": test_days,
        "总交易次数": total_trades,
        "盈利次数": len(win_trades),
        "亏损次数": len(lose_trades),
        "胜率": round(win_rate, 4),
        "盈亏比": round(profit_loss_ratio, 4),
        "总收益（含杠杆）": round(total_profit, 4),
        "年化收益率": round(annual_return, 4),
        "最大回撤": round(max_drawdown, 4),
        "夏普比率": round(sharpe_ratio, 4),
        "卡玛比率": round(calmar_ratio, 4),
        "手续费占比": round(fee_ratio, 4),
        "平均盈利": round(avg_win, 4),
        "平均亏损": round(avg_lose, 4)
    }
    return metrics


# ===================== 5. 主执行函数（一键回测+输出报告） =====================
def main():
    print("=====================================")
    print("🎯 OKX量化策略 完整回测体系")
    print("=====================================")

    # 1. 加载最优参数（从config.json读取，不用手动填）
    print("\n📌 加载调优后的最优参数...")
    best_params = CONFIG["strategy"]
    print("✅ 最优参数加载完成：")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    # 2. 获取历史数据
    df = get_historical_data(limit=DATA_LIMIT)
    if df.empty:
        print("❌ 历史数据获取失败")
        return

    # 3. 执行策略回测
    print("\n🚀 开始执行策略回测...")
    trades_df, train_size = strategy_logic(df, best_params)
    test_df = df.iloc[train_size:].reset_index(drop=True)

    # 4. 计算回测指标
    print("\n📊 计算回测核心指标...")
    metrics = calculate_backtest_metrics(trades_df, test_df)

    # 5. 输出详细回测报告
    print("\n=====================================")
    print("🏆 完整回测报告（核心6大指标已标注）")
    print("=====================================")
    for k, v in metrics.items():
        # 标注核心指标
        if k in ["胜率", "盈亏比", "年化收益率", "最大回撤", "夏普比率", "卡玛比率"]:
            print(f"🔑 {k}: {v}")
        else:
            print(f"  {k}: {v}")

    # 6. 交易记录保存
    if not trades_df.empty:
        trades_df.to_csv("backtest_trades.csv", encoding='utf-8', index=False)
        print(f"\n✅ 交易记录已保存到 backtest_trades.csv")

    # 7. 策略评估结论
    print("\n=====================================")
    print("📝 策略评估结论")
    print("=====================================")
    if metrics.get("夏普比率", 0) >= 1.5 and metrics.get("最大回撤", 0) >= -0.15:
        print("✅ 策略评估：优秀！符合实战要求，可上模拟盘长期运行")
    elif metrics.get("夏普比率", 0) >= 1.0 and metrics.get("最大回撤", 0) >= -0.2:
        print("⚠️  策略评估：合格！需小幅优化参数，降低回撤")
    else:
        print("❌ 策略评估：不合格！建议优化开仓信号或风控参数")


if __name__ == "__main__":
    main()