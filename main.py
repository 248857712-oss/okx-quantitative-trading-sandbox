import logging
import os
import time
import datetime
import pandas as pd
import ta
import requests
import json
import hashlib
import hmac
import base64
from urllib.parse import urlencode, quote
import ccxt

# 导入工具类
from config_utils import load_config
from log_utils import init_logger, trade_logger
from trade_utils import save_trade_record, get_trade_statistics
from gb_stop_loss_take_profit import GBSLTPModel

# 加载配置并校验
config = load_config()
PROXY_SETTINGS = config["proxy"]

# 从usdt_get.py整合的核心函数（保留）
def create_proxy_session():
    session = requests.Session()
    session.proxies = {
        "http": PROXY_SETTINGS["http"],
        "https": PROXY_SETTINGS["https"],
    }
    session.timeout = 15
    requests.packages.urllib3.disable_warnings()
    return session

def test_proxy_connectivity():
    try:
        session = create_proxy_session()
        response = session.get("https://www.okx.com")
        if response.status_code == 200:
            logging.info("✅ 代理连通性测试成功")
            return True
        else:
            logging.error(f"❌ 代理测试失败，状态码: {response.status_code}")
            return False
    except Exception as e:
        logging.error(f"❌ 代理连接失败: {str(e)}")
        return False

def get_okx_sandbox_balance(api_key, api_secret, api_passphrase, is_sim=True):
    try:
        okx = ccxt.okx({
            "apiKey": api_key,
            "secret": api_secret,
            "password": api_passphrase,
            "enableRateLimit": True,
            "sandbox": is_sim,
            "proxies": PROXY_SETTINGS,
            "options": {"defaultType": "swap", "fetchBalance": "all"}
        })
        okx.load_markets()
        balance = okx.fetch_balance()
        total_usdt = float(balance.get('total', {}).get('USDT', 0))
        free_usdt = float(balance.get('free', {}).get('USDT', 0))
        used_usdt = float(balance.get('used', {}).get('USDT', 0))
        logging.info(f"✅ OKX模拟盘USDT余额 | 总计: {total_usdt} | 可用: {free_usdt} | 已用: {used_usdt}")
        return {"total": total_usdt, "free": free_usdt, "used": used_usdt}
    except Exception as e:
        logging.error(f"❌ 余额获取失败: {str(e)}")
        return None

# ================= OKX合约API客户端（真实模拟盘适配） =================
class OKXFuturesAPIClient:
    def __init__(self, api_key, api_secret, passphrase, is_sim=True):
        # CCXT初始化（沙箱模式）
        self.okx_ccxt = ccxt.okx({
            'apiKey': api_key,
            'secret': api_secret,
            'password': passphrase,
            'enableRateLimit': True,
            'sandbox': is_sim,  # 关键：启用沙箱
            'proxies': PROXY_SETTINGS,
            'options': {'defaultType': 'swap'}  # 强制合约类型
        })
        self.okx_ccxt.load_markets()

        # 原生API配置（用于精准控制）
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.base_url = "https://www.okx.com" if is_sim else "https://www.okx.com"
        self.sim = is_sim
        self.session = create_proxy_session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "OK-ACCESS-PASSPHRASE": self.passphrase
        })

    def _get_timestamp(self):
        timestamp_ms = int(time.time() * 1000)
        dt = datetime.datetime.fromtimestamp(timestamp_ms / 1000, datetime.timezone.utc)
        return dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

    def _sign(self, timestamp, method, request_path, query_params=None, body=None):
        query_params = query_params or {}
        body = body or {}
        query_string = '&'.join([f"{k}={v}" for k, v in sorted(query_params.items())]) if query_params else ''
        body_string = json.dumps(body, separators=(',', ':'), ensure_ascii=False) if (body and method == 'POST') else ''
        sign_string = f"{timestamp}{method.upper()}{request_path}{f'?{query_string}' if query_string else ''}{body_string}"
        try:
            sign_bytes = hmac.new(self.api_secret.encode('utf-8'), sign_string.encode('utf-8'), hashlib.sha256).digest()
            return base64.b64encode(sign_bytes).decode('utf-8')
        except Exception as e:
            logging.error(f"签名失败: {str(e)}")
            return ""

    def request(self, method, request_path, params=None, data=None):
        method = method.upper()
        params = params or {}
        data = data or {}
        url = f"{self.base_url}{request_path}"
        timestamp = self._get_timestamp()
        signature = self._sign(timestamp, method, request_path, params, data)
        if not signature:
            logging.error("❌ 签名生成失败")
            return None

        headers = {
            "OK-ACCESS-KEY": self.api_key,
            "OK-ACCESS-SIGN": signature,
            "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json",
            "x-simulated-trading": "1" if self.sim else "0"  # 模拟盘标识
        }

        if method == "GET" and params:
            url += f"?{urlencode(sorted(params.items()))}"

        try:
            if method == "GET":
                response = self.session.get(url, headers=headers, verify=False)
            elif method == "POST":
                response = self.session.post(url, data=json.dumps(data, separators=(',', ':')), headers=headers, verify=False)
            else:
                return None

            response.raise_for_status()
            result = json.loads(response.text)
            if result.get("code") != "0":
                logging.error(f"API错误 | 代码:{result['code']} | 消息:{result['msg']}")
                return None
            return result.get("data", [])
        except Exception as e:
            logging.error(f"请求异常: {str(e)} | URL:{url}")
            return None

    def get_ticker_price(self, symbol):
        try:
            ticker = self.okx_ccxt.fetch_ticker(symbol)
            return float(ticker['last'])
        except Exception as e:
            data = self.request("GET", "/api/v5/market/ticker", params={"instId": symbol, "instType": "SWAP"})
            return float(data[0]['last']) if data else 0.0

    def get_account_balance(self):
        balance_data = get_okx_sandbox_balance(self.api_key, self.api_secret, self.passphrase, self.sim)
        return balance_data["free"] if (balance_data and balance_data["free"] > 0) else 100000.0

    def set_leverage(self, symbol, leverage):
        try:
            # 全仓模式下设置杠杆
            data = {
                "instId": symbol,
                "instType": "SWAP",
                "lever": str(leverage),
                "mgnMode": "cross",  # 全仓
                "posSide": "long"
            }
            self.request("POST", "/api/v5/account/set-leverage", data=data)
            data["posSide"] = "short"
            self.request("POST", "/api/v5/account/set-leverage", data=data)
            logging.info(f"✅ 合约杠杆设置成功 | {leverage}倍（全仓）")
        except Exception as e:
            logging.error(f"杠杆设置失败: {str(e)[:80]}")

    # 新增：真实模拟盘开仓接口
    def open_position(self, symbol, side, amount, price=None):
        """
        真实模拟盘开仓
        side: 'long'/'short'
        amount: 开仓数量（BTC）
        price: 限价单价格，None为市价单
        """
        try:
            data = {
                "instId": symbol,
                "instType": "SWAP",
                "tdMode": "cross",  # 全仓
                "side": "buy" if side == "long" else "sell",
                "posSide": side,
                "ordType": "market" if price is None else "limit",
                "sz": str(amount),
                "px": str(price) if price else None
            }
            # 移除None值
            data = {k: v for k, v in data.items() if v is not None}
            result = self.request("POST", "/api/v5/trade/order", data=data)
            if result:
                order_id = result[0].get('ordId')
                logging.info(f"✅ 【真实模拟盘】{side}开仓成功 | 订单ID:{order_id} | 数量:{amount}")
                return order_id
            return None
        except Exception as e:
            logging.error(f"开仓失败: {str(e)}")
            return None

    # 新增：真实模拟盘平仓接口
    def close_position(self, symbol, side, amount, price=None):
        """
        真实模拟盘平仓
        side: 'long'/'short'
        amount: 平仓数量（BTC）
        price: 限价单价格，None为市价单
        """
        try:
            data = {
                "instId": symbol,
                "instType": "SWAP",
                "tdMode": "cross",
                "side": "sell" if side == "long" else "buy",
                "posSide": side,
                "ordType": "market" if price is None else "limit",
                "sz": str(amount),
                "px": str(price) if price else None,
                "closePos": "true"  # 平仓标识
            }
            data = {k: v for k, v in data.items() if v is not None}
            result = self.request("POST", "/api/v5/trade/order", data=data)
            if result:
                order_id = result[0].get('ordId')
                logging.info(f"✅ 【真实模拟盘】{side}平仓成功 | 订单ID:{order_id} | 数量:{amount}")
                return order_id
            return None
        except Exception as e:
            logging.error(f"平仓失败: {str(e)}")
            return None

# ================= 合约策略主类（真实模拟盘适配） =================
class OKXFuturesTrader:
    def __init__(self, config):
        self.config = config
        self.symbol = config["okx"]["symbol"]  # BTC-USDT-SWAP
        self.leverage = config["strategy"]["leverage"]
        self.position_ratio = config["strategy"]["position_ratio"]
        self.lr_weight = config["strategy"]["lr_weight"]
        self.rf_weight = config["strategy"]["rf_weight"]
        self.vote_threshold = config["strategy"]["vote_threshold"]
        self.tp_prob_threshold = config["strategy"]["tp_prob_threshold"]
        self.sl_prob_threshold = config["strategy"]["sl_prob_threshold"]
        self.cycle_interval = config["strategy"]["cycle_interval"]
        self.boll_window = config["strategy"]["boll_window"]
        self.boll_dev = config["strategy"]["boll_dev"]

        self.min_profit_threshold = 0.001
        self.min_loss_threshold = 0.001
        self.target_profit_ratio = 0.005
        self.min_profit_risk_ratio = 1.5

        # 持仓状态（实时同步平台）
        self.position = 0  # 1多头，-1空头，0无持仓
        self.entry_price = None
        self.hold_amount = 0.0
        self.last_price = 0.0
        self.boll_lower = 0.0
        self.trade_allowed = True

        self.logger = init_logger(config["log"]["log_path"], config["log"]["log_level"])
        self.client = OKXFuturesAPIClient(
            config["okx"]["api_key"],
            config["okx"]["api_secret"],
            config["okx"]["api_passphrase"],
            config["okx"]["is_sim"]
        )
        self.lr = __import__('sklearn.linear_model').linear_model.LogisticRegression(random_state=42, max_iter=200)
        self.rf = __import__('sklearn.ensemble').ensemble.RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        self.sltp_model = GBSLTPModel(random_state=42)
        self.trained = False

        self.timeframe_mapping = {
            '1m': '1M', '5m': '5M', '15m': '15M', '30m': '30M',
            '1h': '1H', '2h': '2H', '4h': '4H', '6h': '6H',
            '12h': '12H', '1d': '1D', '1w': '1W'
        }

        test_proxy_connectivity()
        self.logger.info("✅ 合约策略初始化完成（真实模拟盘模式）")
        self.logger.info(f"交易对: {self.symbol} | 杠杆: {self.leverage}倍 | 仓位比例: {self.position_ratio * 100}%")

    def get_realtime_price(self):
        try:
            realtime_price = self.client.get_ticker_price(self.symbol)
            if realtime_price > 0:
                self.last_price = realtime_price
                self.logger.info(f"✅ 实时价更新 | {self.symbol} = {realtime_price:.2f} USDT")
            return realtime_price
        except Exception as e:
            self.logger.error(f"实时价获取失败: {str(e)[:100]}")
            return self.last_price if self.last_price > 0 else 0.0

    def calculate_order_amount(self, realtime_price=None):
        balance = self.client.get_account_balance()
        if balance <= 0:
            balance = 100000.0
            self.logger.warning(f"⚠️ 使用测试余额 {balance} USDT")

        current_price = realtime_price or self.get_realtime_price()
        if current_price <= 0:
            current_price = 90000.0
            self.logger.warning(f"⚠️ 使用测试价格 {current_price} USDT")

        order_value = balance * self.position_ratio * self.leverage
        order_amount = order_value / current_price
        order_amount = round(order_amount, 2)  # BTC合约最小单位0.01

        min_amount = 0.001
        if order_amount < min_amount:
            self.logger.warning(f"⚠️ 下单量过小，强制设为{min_amount}")
            order_amount = min_amount

        self.logger.info(f"📊 仓位计算 | 余额: {balance} USDT | 下单金额: {order_value:.2f} USDT | 数量: {order_amount} 张")
        return order_amount

    def fetch_ohlcv(self, tf='1h', limit=300):
        try:
            okx_tf = self.timeframe_mapping.get(tf, '1H')
            params = {'instId': self.symbol, 'bar': okx_tf, 'limit': limit, 'instType': 'SWAP'}
            data = self.client.request("GET", "/api/v5/market/history-candles", params=params)
            if not data:
                self.logger.error("❌ K线数据为空")
                return pd.DataFrame()

            df = pd.DataFrame(data, columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'volCcy', 'volCcyQuote', 'confirm'])
            for col in ['open', 'high', 'low', 'close', 'vol']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df['ts'] = pd.to_datetime(pd.to_numeric(df['ts']), unit='ms')
            df = df.dropna().sort_values('ts').reset_index(drop=True)
            self.logger.info(f"✅ 获取K线 | 周期:{tf} | 数量:{len(df)}")
            return df
        except Exception as e:
            self.logger.error(f"K线获取失败: {str(e)[:100]}")
            return pd.DataFrame()

    @trade_logger
    def train(self, df):
        if df.empty:
            self.logger.error("❌ 训练数据为空")
            return
        df = df.copy()
        df['ret'] = (df['close'] - df['open']) / df['open']
        df['range'] = (df['high'] - df['low']) / df['open']
        df['vol_ratio'] = df['vol'] / df['vol'].rolling(20).mean()
        df['y'] = (df['close'].shift(-1) > df['close']).astype(int)
        df = df.dropna()
        if len(df) < 10:
            self.logger.error(f"❌ 样本不足({len(df)}条)")
            return
        X = df[['ret', 'range', 'vol_ratio']]
        y = df['y']
        self.lr.fit(X, y)
        self.rf.fit(X, y)
        self.sltp_model.train(df, tp_threshold=0.002, sl_threshold=0.002)
        self.trained = True
        self.logger.info(f"✅ 模型训练完成 | 上涨概率:{y.mean():.2%}")

    def signal(self, df):
        if not self.trained or df.empty or len(df) < 20:
            return 0
        try:
            latest = df.iloc[-1]
            vol_mean = df['vol'].rolling(20).mean().iloc[-1]
            ret = (latest['close'] - latest['open']) / latest['open'] if latest['open'] != 0 else 0
            range_val = (latest['high'] - latest['low']) / latest['open'] if latest['open'] != 0 else 0
            vol_ratio = latest['vol'] / vol_mean if vol_mean > 0 else 1.0
            x = pd.DataFrame([{'ret': ret, 'range': range_val, 'vol_ratio': vol_ratio}])
            lr_prob = self.lr.predict_proba(x)[0][1]
            rf_prob = self.rf.predict_proba(x)[0][1]
            weighted_prob = lr_prob * self.lr_weight + rf_prob * self.rf_weight
            self.logger.info(f"📊 信号 | LR:{lr_prob:.2%} | RF:{rf_prob:.2%} | 加权:{weighted_prob:.2%}")
            return 1 if weighted_prob >= self.vote_threshold else 0
        except Exception as e:
            self.logger.error(f"信号计算失败: {str(e)[:100]}")
            return 0

    def calculate_profit_risk_ratio(self, df):
        if df.empty or len(df) < 10:
            return 0, 0
        recent_df = df.iloc[-10:]
        avg_range = (recent_df['high'] - recent_df['low']).mean() / recent_df['close'].mean()
        potential_profit = self.target_profit_ratio
        potential_risk = avg_range / 2
        profit_risk_ratio = potential_profit / potential_risk if potential_risk > 0 else 0
        return profit_risk_ratio, potential_profit

    def check_pre_open_sltp(self, df):
        if not self.trained or df.empty:
            return True
        pre_entry_price = self.get_realtime_price() or df['close'].iloc[-1]
        tp_prob, sl_prob = self.sltp_model.predict(df, entry_price=pre_entry_price)
        prob_check = tp_prob >= self.tp_prob_threshold or sl_prob >= self.sl_prob_threshold
        profit_risk_ratio, potential_profit = self.calculate_profit_risk_ratio(df)
        profit_check = (profit_risk_ratio >= self.min_profit_risk_ratio) and (potential_profit >= self.target_profit_ratio)
        self.logger.info(f"📊 开仓检查 | TP概率:{tp_prob:.2%} | SL概率:{sl_prob:.2%} | 收益风险比:{profit_risk_ratio:.2f}")
        return prob_check and profit_check

    def check_stop_loss_take_profit(self, df):
        if self.position == 0 or self.entry_price is None:
            return
        realtime_price = self.get_realtime_price()
        if realtime_price <= 0:
            self.logger.warning("⚠️ 实时价无效，跳过止盈止损")
            return

        if self.boll_lower > 0 and realtime_price < self.boll_lower:
            self.logger.info(f"🛑 跌破布林下轨强制平仓 | 当前价:{realtime_price:.2f} < 下轨:{self.boll_lower:.2f}")
            self.close_position(is_force=True)
            return

        # 计算盈亏
        if self.position == 1:
            profit_ratio = (realtime_price - self.entry_price) / self.entry_price
        else:
            profit_ratio = (self.entry_price - realtime_price) / self.entry_price

        profit_abs = profit_ratio * self.hold_amount * self.entry_price
        profit_status = "盈利" if profit_ratio > 0 else "亏损" if profit_ratio < 0 else "持平"
        tp_prob, sl_prob = self.sltp_model.predict(df, entry_price=self.entry_price)

        tp_conditions = [
            profit_ratio >= self.min_profit_threshold,
            profit_ratio >= self.target_profit_ratio,
            tp_prob >= self.tp_prob_threshold
        ]
        sl_conditions = [
            profit_ratio <= -self.min_loss_threshold,
            sl_prob >= self.sl_prob_threshold
        ]

        self.logger.info(f"📊 盈亏状态 | {profit_status} {profit_ratio:.2%} | 盈亏金额: {profit_abs:.2f} USDT")
        if all(tp_conditions):
            self.logger.info(f"🚀 触发止盈 | 盈利{profit_ratio:.2%} ≥ 目标{self.target_profit_ratio * 100}%")
            self.close_position(is_force=True)
        elif all(sl_conditions):
            self.logger.info(f"🛑 触发止损 | 亏损{abs(profit_ratio):.2%} ≥ {self.min_loss_threshold * 100}%")
            self.close_position(is_force=True)

    def boll_filter(self):
        try:
            df = self.fetch_ohlcv('1d', 50)
            if df.empty:
                self.trade_allowed = True
                return
            bb = ta.volatility.BollingerBands(df['close'], window=self.boll_window, window_dev=self.boll_dev)
            boll_low = bb.bollinger_lband().iloc[-1]
            self.boll_lower = boll_low
            current_price = self.get_realtime_price()
            self.trade_allowed = current_price > boll_low
            status = "✅ 允许交易" if self.trade_allowed else "⚠️ 禁止交易（跌破布林下轨）"
            self.logger.info(f"{status} | 当前价:{current_price:.2f} | 布林下轨:{boll_low:.2f}")
        except Exception as e:
            self.logger.error(f"布林带过滤失败: {str(e)[:100]}")
            self.trade_allowed = True

    def check_position(self):
        """实时同步平台持仓状态"""
        try:
            params = {
                "instType": "SWAP",
                "instId": self.symbol,
                "mgnMode": "cross"
            }
            result = self.client.request("GET", "/api/v5/account/positions", params=params)
            if result and len(result) > 0:
                for pos in result:
                    pos_side = pos.get('posSide')
                    hold_amount = float(pos.get('pos', 0))
                    if hold_amount > 0:
                        self.position = 1 if pos_side == "long" else -1
                        self.entry_price = float(pos.get('avgPx', 0))
                        self.hold_amount = hold_amount
                        self.logger.info(f"✅ 当前持仓 | {pos_side} {hold_amount:.3f} BTC | 开仓价:{self.entry_price:.2f}")
                        return

            self.position = 0
            self.entry_price = None
            self.hold_amount = 0
            self.logger.info("✅ 当前无持仓")
        except Exception as e:
            self.logger.warning(f"持仓查询失败: {str(e)[:50]}")

    @trade_logger
    def open_long(self):
        if self.position != 0:
            self.logger.warning("⚠️ 当前已有持仓，无法开多")
            return None

        realtime_price = self.get_realtime_price()
        if realtime_price <= 0:
            self.logger.error("❌ 实时价无效，无法开多")
            return None

        order_amount = self.calculate_order_amount(realtime_price)
        if order_amount <= 0:
            return None

        # 真实模拟盘开多
        order_id = self.client.open_position(self.symbol, 'long', order_amount)
        if order_id:
            self.position = 1
            self.entry_price = realtime_price
            self.hold_amount = order_amount
            self.logger.info(f"🟢 【真实模拟盘】开多成功 | 订单ID:{order_id} | 均价:{self.entry_price:.2f} | 数量:{order_amount} BTC")
            # 保存记录
            trade_record = {
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "type": "【真实模拟盘】合约开多",
                "price": self.entry_price,
                "size": order_amount,
                "profit": 0.0,
                "order_id": order_id
            }
            save_trade_record(trade_record)
            return trade_record
        return None

    @trade_logger
    def close_position(self, is_force=False):
        if not is_force and self.position == 0:
            self.logger.warning("⚠️ 当前无持仓，无法平仓")
            return None

        realtime_price = self.get_realtime_price()
        if realtime_price <= 0:
            self.logger.error("❌ 实时价无效，无法平仓")
            return None

        order_amount = self.hold_amount if self.hold_amount > 0 else self.calculate_order_amount(realtime_price)
        if order_amount <= 0:
            return None

        # 真实模拟盘平仓
        side = 'long' if self.position == 1 else 'short'
        order_id = self.client.close_position(self.symbol, side, order_amount)
        if order_id:
            # 计算盈亏
            if self.position == 1:
                profit = (realtime_price - self.entry_price) * order_amount
                trade_type = "【真实模拟盘】合约平多"
            else:
                profit = (self.entry_price - realtime_price) * order_amount
                trade_type = "【真实模拟盘】合约平空"

            # 更新状态
            self.position = 0
            self.entry_price = None
            self.hold_amount = 0
            self.logger.info(f"🔴 {trade_type}成功 | 订单ID:{order_id} | 均价:{realtime_price:.2f} | 盈亏:{profit:.2f} USDT")
            # 保存记录
            trade_record = {
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "type": trade_type,
                "price": realtime_price,
                "size": order_amount,
                "profit": profit,
                "order_id": order_id
            }
            save_trade_record(trade_record)
            return trade_record
        return None

    def force_close_position(self):
        self.logger.info("\n🔴 执行强制平仓...")
        max_retry = 3
        retry_count = 0
        while retry_count < max_retry and self.position != 0:
            self.logger.info(f"📌 平仓重试 {retry_count + 1}/{max_retry}")
            result = self.close_position(is_force=True)
            if result:
                self.logger.info("✅ 强制平仓成功")
                break
            retry_count += 1
            time.sleep(2)
        if self.position != 0:
            self.logger.error("❌ 强制平仓失败")

    def run_strategy(self):
        self.logger.info(f"{'=' * 60}")
        self.logger.info(f"📈 OKX真实模拟盘合约策略启动 | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"{'=' * 60}\n")
        # 初始化
        self.client.set_leverage(self.symbol, self.leverage)
        self.check_position()
        df_init = self.fetch_ohlcv('1h', 500)
        self.train(df_init)
        cycle = 1
        total_trades = 0
        try:
            while True:
                try:
                    self.logger.info(f"\n{'-' * 40} 第{cycle}轮循环 {'-' * 40}")
                    self.client.get_account_balance()
                    self.get_realtime_price()
                    df = self.fetch_ohlcv('1h', 300)
                    self.boll_filter()
                    self.check_position()  # 每次循环同步持仓
                    # 定期重训
                    if cycle % 24 == 0 and not df.empty:
                        self.logger.info("🔄 定期重训模型...")
                        self.train(df)
                    # 交易信号
                    signal = self.signal(df)
                    self.logger.info(f"📊 交易信号:{'📈 开多' if signal == 1 else '📉 观望'} | 持仓:{self.position}")
                    # 执行交易
                    trade_executed = False
                    if self.trade_allowed:
                        if signal == 1 and self.position == 0:
                            if self.check_pre_open_sltp(df):
                                result = self.open_long()
                                trade_executed = result is not None
                            else:
                                self.logger.info("📉 开仓检查不达标，放弃开多")
                        if self.position != 0 and not df.empty:
                            self.check_stop_loss_take_profit(df)
                    if trade_executed:
                        total_trades += 1
                    self.logger.info(f"📋 运行统计 | 总交易次数:{total_trades}")
                    cycle += 1
                    self.logger.info(f"⏳ 等待{self.cycle_interval}秒...")
                    time.sleep(self.cycle_interval)
                except Exception as e:
                    self.logger.error(f"循环异常: {str(e)[:80]}")
                    time.sleep(30)
        except KeyboardInterrupt:
            self.logger.info("\n🛑 用户终止程序")
            if self.position != 0:
                self.force_close_position()
            # 输出统计
            self.logger.info(f"\n{'=' * 50} 交易统计报告 {'=' * 50}")
            stats = get_trade_statistics()
            if not stats.empty:
                self.logger.info(f"\n{stats.to_string(index=False)}")
            else:
                self.logger.info("📊 暂无交易记录")
            self.logger.info(f"📋 策略结束 | 总交易次数:{total_trades} | 时间:{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info(f"{'=' * 100}")

if __name__ == "__main__":
    trader = OKXFuturesTrader(config)
    trader.run_strategy()