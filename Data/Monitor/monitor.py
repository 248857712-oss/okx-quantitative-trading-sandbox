import streamlit as st
import pandas as pd
import plotly.express as px
import os
import time
import json  # 新增：读取JSON格式的交易记录
from datetime import datetime

# 页面配置
st.set_page_config(
    page_title="OKX合约量化策略监控面板",
    page_icon="📈",
    layout="wide"
)

st.title("📈 OKX 合约模拟盘量化策略实时监控面板")


# ================= 修复1：实现兼容的交易记录读取函数 =================
def load_trade_records(record_path: str = "./trade_records.json") -> pd.DataFrame:
    """读取JSON格式的交易记录（适配合约策略）"""
    # 初始化空DataFrame
    df = pd.DataFrame(columns=["time", "type", "price", "size", "profit", "order_id"])

    if not os.path.exists(record_path):
        return df

    try:
        with open(record_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        # 转换为DataFrame并处理格式
        df = pd.DataFrame(records)
        if not df.empty:
            # 统一字段格式（适配合约策略的开多/平仓）
            df["type"] = df["type"].replace({
                "【模拟】合约开多": "开多",
                "【模拟】合约平多": "平仓",
                "【模拟】合约平空": "平仓"
            })
            # 转换时间格式
            df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M:%S")
            # 数值类型转换
            df["price"] = pd.to_numeric(df["price"], errors="coerce")
            df["size"] = pd.to_numeric(df["size"], errors="coerce")
            df["profit"] = pd.to_numeric(df["profit"], errors="coerce")
    except Exception as e:
        st.warning(f"读取交易记录失败：{str(e)}")

    return df


# ================= 1. 读取交易记录 =================
trades_df = load_trade_records("./trade_records.json")  # 适配JSON格式

# 分两列展示
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 交易记录")
    if trades_df.empty:
        st.info("暂无交易记录，请先运行策略")
    else:
        # 显示表格（修复字段适配问题）
        st.dataframe(
            trades_df,
            column_config={
                "time": st.column_config.DatetimeColumn("时间", format="YYYY-MM-DD HH:mm:ss"),
                "type": st.column_config.SelectboxColumn(
                    "交易类型",
                    options=["开多", "平仓"],  # 适配合约策略
                    default="开多"
                ),
                "price": st.column_config.NumberColumn("价格 (USDT)", format="%.2f"),
                "size": st.column_config.NumberColumn("持仓数量 (BTC)", format="%.3f"),  # 合约精度3位
                "profit": st.column_config.NumberColumn("盈亏 (USDT)", format="%.2f"),
                "order_id": "订单ID"
            },
            use_container_width=True,
            hide_index=True  # 隐藏索引更美观
        )

with col2:
    st.subheader("💰 盈亏统计")
    if trades_df.empty:
        total_profit = 0.0
        trade_count = 0
        max_profit = 0.0
        max_loss = 0.0
    else:
        total_profit = trades_df["profit"].sum()
        trade_count = len(trades_df)
        max_profit = trades_df["profit"].max() if trade_count > 0 else 0.0
        max_loss = trades_df["profit"].min() if trade_count > 0 else 0.0

    # 显示关键指标（修复空值报错）
    st.metric("累计盈亏", f"{total_profit:.2f} USDT")
    st.metric("总交易次数", trade_count)
    st.metric("单笔最大盈利", f"{max_profit:.2f} USDT")
    st.metric("单笔最大亏损", f"{max_loss:.2f} USDT")

    # 新增：胜率计算（仅统计平仓记录）
    if trade_count > 0:
        close_trades = trades_df[trades_df["type"] == "平仓"]
        win_trades = len(close_trades[close_trades["profit"] > 0])
        win_rate = (win_trades / len(close_trades) * 100) if len(close_trades) > 0 else 0
        st.metric("胜率", f"{win_rate:.1f}%")

# ================= 2. 盈亏走势图 =================
st.subheader("📊 盈亏走势")
if not trades_df.empty:
    # 按时间排序
    trades_df = trades_df.sort_values("time")
    # 只统计平仓的盈亏（开多无盈亏）
    close_trades = trades_df[trades_df["type"] == "平仓"].copy()
    if not close_trades.empty:
        # 计算累计盈亏
        close_trades["累计盈亏"] = close_trades["profit"].cumsum()

        # 画图（修复空值问题）
        fig = px.line(
            close_trades,
            x="time",
            y="累计盈亏",
            title="累计盈亏变化曲线",
            labels={"time": "时间", "累计盈亏": "累计盈亏 (USDT)"},
            line_shape="spline",
            color_discrete_sequence=["#1E90FF"],
            markers=True  # 新增：显示数据点
        )
        fig.update_layout(
            height=400,
            xaxis_title="交易时间",
            yaxis_title="累计盈亏 (USDT)",
            title_x=0.5  # 标题居中
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("暂无平仓记录，无法生成盈亏走势")
else:
    st.info("暂无交易数据，无法生成走势图")

# ================= 3. 最新运行日志 =================
st.subheader("📜 最新运行日志")
log_dir = "./logs"
if os.path.exists(log_dir):
    # 匹配合约策略的日志文件名
    log_files = [f for f in os.listdir(log_dir) if f.startswith("okx_spot_strategy_") and f.endswith(".log")]
    if log_files:
        latest_log_file = max(log_files, key=lambda x: os.path.getmtime(os.path.join(log_dir, x)))
        log_path = os.path.join(log_dir, latest_log_file)

        # 读取最后 2000 字符（避免加载过大）
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                log_content = f.read()[-2000:]  # 取最后2000字符
            # 美化日志显示
            st.text_area(
                f"日志文件：{latest_log_file}",
                log_content,
                height=300,
                placeholder="暂无日志内容"
            )
        except Exception as e:
            st.error(f"读取日志失败：{str(e)}")
    else:
        st.info("暂无日志文件，请先运行策略")
else:
    st.info("日志目录不存在，策略运行后会自动创建")

# ================= 4. 自动刷新 =================
st.sidebar.header("🔧 刷新设置")
st.sidebar.button("🔄 手动刷新", on_click=lambda: st.rerun())
refresh_interval = st.sidebar.slider("自动刷新间隔（秒）", 10, 60, 30)  # 新增：可调节间隔
st_autorefresh = st.sidebar.checkbox("开启自动刷新", value=True)

if st_autorefresh:
    # 修复：避免阻塞，用streamlit的timer更友好
    time.sleep(refresh_interval)
    st.rerun()