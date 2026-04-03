import json
import os
from datetime import datetime
import pandas as pd


def save_trade_record(record: dict, record_path: str = "./trade_records.json"):
    """保存交易记录到JSON文件，支持追加写入"""
    # 确保记录包含时间戳
    record["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 读取已有记录
    records = []
    if os.path.exists(record_path):
        with open(record_path, "r", encoding="utf-8") as f:
            try:
                records = json.load(f)
            except json.JSONDecodeError:
                pass  # 文件损坏时重新创建

    # 追加新记录
    records.append(record)

    # 写入文件
    with open(record_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=4, ensure_ascii=False)


def get_trade_statistics(record_path: str = "./trade_records.json") -> pd.DataFrame:
    """生成交易统计报告：总收益、胜率、平均盈亏等"""
    if not os.path.exists(record_path):
        return pd.DataFrame()

    with open(record_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    df = pd.DataFrame(records)
    if df.empty:
        return df

    # 计算关键统计指标
    df["profit"] = df["profit"].astype(float)
    total_trades = len(df)
    winning_trades = len(df[df["profit"] > 0])
    losing_trades = len(df[df["profit"] < 0])
    win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
    total_profit = df["profit"].sum()
    avg_profit = df["profit"].mean() if total_trades > 0 else 0

    # 生成统计DataFrame
    stats = pd.DataFrame({
        "统计项": ["总交易次数", "盈利次数", "亏损次数", "胜率(%)", "总收益(USDT)", "平均单次收益(USDT)"],
        "数值": [total_trades, winning_trades, losing_trades, round(win_rate, 2), round(total_profit, 2),
                 round(avg_profit, 2)]
    })

    return stats