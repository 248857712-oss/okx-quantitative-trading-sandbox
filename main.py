import pandas as pd
import ta
import time
import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import requests
import json
import hashlib
import hmac
import base64
from urllib.parse import urlencode, quote
# 导入外部工具
from config_utils import load_config
from log_utils import init_logger
from trade_utils import save_trade_record
from gb_stop_loss_take_profit import GBSLTPModel
# ========== 新增：导入1.py的余额获取函数 ==========
from usdt_get import get_okx_sandbox_balance  # 注意：1.py需重命名为py1.py（避免数字开头），或按实际文件名调整

# ================= 代理配置 =================
PROXY_SETTINGS = {
    'http': 'http://127.0.0.1:10808',
    'https': 'http://127.0.0.1:10808'
}


# ================= OKX官方标准签名类 =================
class OKXAPIClient:
    def __init__(self, api_key, api_secret, passphrase, is_sim=True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.base_url = "https://www.okx.com" if is_sim else "https://www.okx.com"
        self.sim = is_sim

        # 初始化session
        self.session = requests.Session()
        self.session.proxies = PROXY_SETTINGS
        requests.packages.urllib3.disable_warnings()

    def _get_timestamp(self):
        now = datetime.datetime.now(datetime.timezone.utc)
        timestamp = now.isoformat(timespec='milliseconds').replace('+00:00', 'Z')
        return timestamp

    def _sign(self, timestamp, method, request_path, body="", ensure_ascii=True):
        if isinstance(body, dict):
            body = json.dumps(body, separators=(',', ':')) if body else ""
        message = f"{timestamp}{method}{request_path}{body}"
        mac = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        )
        signature = base64.b64encode(mac.digest()).decode('utf-8')
        return signature

    def request(self, method, request_path, params=None, data=None):
        timestamp = self._get_timestamp()
        method = method.upper()
        url = f"{self.base_url}{request_path}"
        params = params or {}
        data = data or {}

        if method == "GET" and params:
            encoded_params = urlencode(params, safe='=&')
            url += "?" + encoded_params
            body = ""
        elif method == "POST" and data:
            body = json.dumps(data, separators=(',', ':'))
        else:
            body = ""

        signature = self._sign(timestamp, method, request_path, body)
        headers = {
            "OK-ACCESS-KEY": self.api_key,
            "OK-ACCESS-SIGN": signature,
            "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json",
            "User-Agent": "okx-python-sdk/1.0.0"
        }
        if self.sim:
            headers["x-simulated-trading"] = "1"

        try:
            if method == "GET":
                response = self.session.get(url, headers=headers, timeout=20, verify=False)
            elif method == "POST":
                response = self.session.post(url, data=body, headers=headers, timeout=20, verify=False)
            else:
                return None
            response.raise_for_status()
            result = response.json()
            if result["code"] == "0":
                return result["data"]
            else:
                return None
        except Exception as e:
            return None

    # ========== 核心修改：替换余额获取逻辑 ==========
    def get_account_balance(self):
        try:
            # 调用1.py的函数获取模拟盘余额
            balance_data = get_okx_sandbox_balance()
            if not balance_data:
                print("❌ 从1.py获取余额失败")
                return 0.0

            # 提取USDT余额（和1.py的解析逻辑保持一致）
            total_usdt = balance_data.get('total', {}).get('USDT', 0)
            avail_usdt = balance_data.get('free', {}).get('USDT', 0)  # 可用余额（可选：也可改用总计）

            # 打印调试信息
            print(f"✅ 从1.py获取余额 | 总计USDT: {total_usdt} | 可用USDT: {avail_usdt}")
            # 策略中建议使用可用余额计算仓位
            return float(avail_usdt) if float(avail_usdt) > 0 else 0.0

        except Exception as e:
            print(f"❌ 解析1.py余额数据错误: {str(e)}")
            return 0.0

    # 新增：获取合约最新价格
    def get_ticker_price(self, symbol):
        data = self.request("GET", "/api/v5/market/ticker", params={"instId": symbol})
        if not data:
            return 0.0
        return float(data[0]['last'])


# ================= 量化策略主类 =================
class OKXSimFuturesTrader:
    def __init__(self, config):
        self.config = config
        # 从配置读取参数
        self.symbol = config["okx"]["symbol"]
        self.leverage = config["strategy"]["leverage"]
        self.position_ratio = config["strategy"]["position_ratio"]
        self.lr_weight = config["strategy"]["lr_weight"]
        self.rf_weight = config["strategy"]["rf_weight"]
        self.vote_threshold = config["strategy"]["vote_threshold"]
        self.tp_prob_threshold = config["strategy"]["tp_prob_threshold"]
        self.sl_prob_threshold = config["strategy"]["sl_prob_threshold"]
        self.cycle_interval = config["strategy"]["cycle_interval"]
        # 新增：最小盈亏阈值（避免微小波动触发）
        self.min_profit_threshold = 0.001  # 0.1%最小盈利才触发止盈
        self.min_loss_threshold = 0.001  # 0.1%最小亏损才触发止损

        # 持仓状态
        self.position = 0
        self.entry_price = None
        self.last_price = None

        # 初始化日志
        self.logger = init_logger(
            log_path=config["log"]["log_path"],
            log_level=config["log"]["log_level"]
        )

        # 初始化OKX客户端
        self.client = OKXAPIClient(
            api_key=config["okx"]["api_key"],
            api_secret=config["okx"]["api_secret"],
            passphrase=config["okx"]["api_passphrase"],
            is_sim=config["okx"]["is_sim"]
        )

        # 初始化模型
        self.lr = LogisticRegression(random_state=42, max_iter=200)
        self.rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        self.sltp_model = GBSLTPModel(random_state=42)

        self.trained = False
        self.trade_allowed = True

        # 时间周期映射
        self.timeframe_mapping = {
            '1m': '1M', '5m': '5M', '15m': '15M', '30m': '30M',
            '1h': '1H', '2h': '2H', '4h': '4H', '6h': '6H',
            '12h': '12H', '1d': '1D', '1w': '1W'
        }

        # 日志初始化信息
        self.logger.info("✅ 交易所初始化成功（合约模拟盘）")
        self.logger.info(f"✅ 模型配置: 逻辑回归(权重{self.lr_weight}) + 随机森林(权重{self.rf_weight})")
        self.logger.info(f"✅ 仓位配置: 账户权益的{self.position_ratio * 100}%")
        self.logger.info(
            f"✅ 止盈止损阈值: 止盈概率≥{self.tp_prob_threshold * 100}% | 止损概率≥{self.sl_prob_threshold * 100}%")
        self.logger.info(
            f"✅ 最小盈亏阈值: 盈利≥{self.min_profit_threshold * 100}% | 亏损≥{self.min_loss_threshold * 100}%")
        self.logger.warning("⚠️  按下Ctrl+C将自动平仓并退出程序")

    # ================= 辅助函数：计算下单张数 =================
    def calculate_order_size(self):
        """计算下单张数 = (账户余额 * 仓位比例 * 杠杆) / 最新价格"""
        # 获取账户余额
        balance = self.client.get_account_balance()
        if balance <= 0:
            self.logger.error("❌ 账户余额不足，无法计算仓位")
            return 0.0
        # 获取最新价格
        self.last_price = self.client.get_ticker_price(self.symbol)
        if self.last_price <= 0:
            self.logger.error("❌ 无法获取最新价格，无法计算仓位")
            return 0.0
        # 计算下单张数
        order_size = (balance * self.position_ratio * self.leverage) / self.last_price
        # 修正：按OKX BTC合约规则，保留3位小数（最小0.001张）
        order_size = round(order_size, 2)
        # 确保最小下单量
        if order_size < 0.001:
            self.logger.warning(f"⚠️  计算的下单量{order_size}小于最小0.001张，强制设为0.001")
            order_size = 0.001
        self.logger.info(
            f"📊 仓位计算 | 账户余额: {balance} USDT | 最新价: {self.last_price} USDT | 下单张数: {order_size}")
        return order_size

    def fetch_ohlcv(self, tf='1h', limit=300):
        try:
            okx_tf = self.timeframe_mapping.get(tf, '1H')
            params = {'instId': self.symbol, 'bar': okx_tf, 'limit': limit}
            data = self.client.request("GET", "/api/v5/market/history-candles", params=params)
            if not data:
                self.logger.error(f"❌ K线获取失败: 无返回数据")
                return pd.DataFrame()

            df = pd.DataFrame(data, columns=[
                'ts', 'open', 'high', 'low', 'close', 'vol', 'volCcy', 'volCcyQuote', 'confirm'
            ])
            # 数据格式转换
            for col in ['open', 'high', 'low', 'close', 'vol']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df['ts'] = pd.to_datetime(pd.to_numeric(df['ts']), unit='ms')
            df = df.dropna().sort_values('ts').reset_index(drop=True)

            self.last_price = df['close'].iloc[-1]
            self.logger.info(f"✅ 获取K线 | 周期:{tf} | 数量:{len(df)} | 最新价:{self.last_price:.2f}")
            return df
        except Exception as e:
            self.logger.error(f"❌ K线获取失败: {str(e)[:100]}")
            return pd.DataFrame()

    def train(self, df):
        if df.empty:
            self.logger.error("❌ 训练数据为空")
            return
        df = df.copy()
        # 提取特征
        df['ret'] = (df['close'] - df['open']) / df['open']
        df['range'] = (df['high'] - df['low']) / df['open']
        df['vol_ratio'] = df['vol'] / df['vol'].rolling(20).mean()
        df['y'] = (df['close'].shift(-1) > df['close']).astype(int)
        df = df.dropna()

        if len(df) < 10:
            self.logger.error(f"❌ 训练样本不足({len(df)}条)")
            return

        X = df[['ret', 'range', 'vol_ratio']]
        y = df['y']
        # 训练模型
        self.lr.fit(X, y)
        self.rf.fit(X, y)
        # 训练止盈止损模型（核心修改：降低阈值到0.2%，提升概率敏感度）
        self.sltp_model.train(df, tp_threshold=0.002, sl_threshold=0.002)

        self.trained = True
        self.logger.info(f"✅ 交易信号模型训练完成 | 样本数:{len(df)} | 上涨概率:{y.mean():.2%}")

    def signal(self, df):
        if not self.trained or df.empty or len(df) < 20:
            return 0
        try:
            latest = df.iloc[-1]
            vol_mean = df['vol'].rolling(20).mean().iloc[-1]
            # 提取特征
            ret = (latest['close'] - latest['open']) / latest['open'] if latest['open'] != 0 else 0
            range_val = (latest['high'] - latest['low']) / latest['open'] if latest['open'] != 0 else 0
            vol_ratio = latest['vol'] / vol_mean if vol_mean > 0 else 1.0
            x = pd.DataFrame([{'ret': ret, 'range': range_val, 'vol_ratio': vol_ratio}])

            # 预测概率（正类概率）
            lr_prob = self.lr.predict_proba(x)[0][1]
            rf_prob = self.rf.predict_proba(x)[0][1]
            # 加权投票
            weighted_prob = lr_prob * self.lr_weight + rf_prob * self.rf_weight
            self.logger.info(f"📊 信号计算 | LR概率:{lr_prob:.2%} | RF概率:{rf_prob:.2%} | 加权概率:{weighted_prob:.2%}")
            return 1 if weighted_prob >= self.vote_threshold else 0
        except Exception as e:
            self.logger.error(f"❌ 信号计算失败: {str(e)[:100]}")
            return 0

    # ========== 新增：开仓前止盈止损概率检查 ==========
    def check_pre_open_sltp(self, df):
        """
        开仓前检查止盈止损概率是否低于阈值
        :param df: K线数据
        :return: bool: True=允许开仓，False=禁止开仓
        """
        if not self.trained or df.empty:
            self.logger.warning("⚠️  模型未训练或K线数据为空，跳过开仓前SLTP检查")
            return True  # 兜底：无数据时允许开仓

        # 模拟开仓价为当前最新价（开仓前无真实开仓价）
        pre_entry_price = self.last_price if self.last_price else df['close'].iloc[-1]
        tp_prob, sl_prob = self.sltp_model.predict(df, entry_price=pre_entry_price, debug=True)

        self.logger.info(
            f"📊 开仓前SLTP检查 | 止盈概率:{tp_prob:.2%} | 止损概率:{sl_prob:.2%} | 阈值:{self.tp_prob_threshold * 100}%/{self.sl_prob_threshold * 100}%")

        # 核心条件：止盈概率 < 止盈阈值 且 止损概率 < 止损阈值
        if tp_prob < self.tp_prob_threshold and sl_prob < self.sl_prob_threshold:
            self.logger.info("✅ 开仓前SLTP检查通过：概率低于阈值")
            return True
        else:
            self.logger.warning(f"❌ 开仓前SLTP检查失败：概率达到或超过阈值，禁止开仓")
            return False

    # ========== 核心修复：止盈止损结合实际盈亏状态 ==========
    def check_stop_loss_take_profit(self, df):
        """
        止盈止损检查：结合实际盈亏状态 + 概率阈值
        - 止盈：盈利 ≥ 最小盈利阈值 且 止盈概率 ≥ 阈值
        - 止损：亏损 ≥ 最小亏损阈值 且 止损概率 ≥ 阈值
        """
        if self.position != 1:
            return
        if self.entry_price is None or self.last_price is None:
            self.logger.warning("⚠️  无开仓价/最新价，止盈止损预测跳过")
            return

        # 1. 计算实际盈亏比例
        profit_ratio = (self.last_price - self.entry_price) / self.entry_price
        profit_abs = self.last_price - self.entry_price  # 绝对盈亏
        profit_status = "盈利" if profit_ratio > 0 else "亏损" if profit_ratio < 0 else "持平"

        # 2. 预测止盈止损概率
        tp_prob, sl_prob = self.sltp_model.predict(df, entry_price=self.entry_price, debug=True)
        self.logger.info(
            f"📊 止盈止损预测 | 当前{profit_status} {profit_ratio:.2%} (¥{profit_abs:.2f}) | "
            f"止盈概率:{tp_prob:.2%} | 止损概率:{sl_prob:.2%}"
        )

        # 3. 止盈触发：仅当盈利且达到最小盈利阈值 + 止盈概率达标
        if profit_ratio >= self.min_profit_threshold and tp_prob >= self.tp_prob_threshold:
            self.logger.info(
                f"🚀 触发止盈 | 盈利{profit_ratio:.2%} ≥ {self.min_profit_threshold * 100}% + "
                f"止盈概率{tp_prob:.2%} ≥ {self.tp_prob_threshold * 100}%"
            )
            self.sell(is_force=True)

        # 4. 止损触发：仅当亏损且达到最小亏损阈值 + 止损概率达标
        elif profit_ratio <= -self.min_loss_threshold and sl_prob >= self.sl_prob_threshold:
            self.logger.info(
                f"🛑 触发止损 | 亏损{abs(profit_ratio):.2%} ≥ {self.min_loss_threshold * 100}% + "
                f"止损概率{sl_prob:.2%} ≥ {self.sl_prob_threshold * 100}%"
            )
            self.sell(is_force=True)

        # 5. 无触发情况
        else:
            reasons = []
            if profit_ratio < self.min_profit_threshold and tp_prob >= self.tp_prob_threshold:
                reasons.append(f"盈利不足({profit_ratio:.2%} < {self.min_profit_threshold * 100}%)")
            if profit_ratio > -self.min_loss_threshold and sl_prob >= self.sl_prob_threshold:
                reasons.append(f"亏损不足({abs(profit_ratio):.2%} < {self.min_loss_threshold * 100}%)")
            if tp_prob < self.tp_prob_threshold and sl_prob < self.sl_prob_threshold:
                reasons.append("概率未达标")

            if reasons:
                self.logger.info(f"ℹ️  未触发止盈止损 | 原因: {'; '.join(reasons)}")

    def boll_filter(self):
        try:
            df = self.fetch_ohlcv('1d', 50)
            if df.empty:
                self.trade_allowed = True
                return
            bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=1)
            boll_low = bb.bollinger_lband().iloc[-1]
            current_price = df['close'].iloc[-1]
            self.trade_allowed = current_price > boll_low
            status = "✅ 允许交易" if self.trade_allowed else "⚠️  禁止交易"
            self.logger.info(f"{status} | 当前价:{current_price:.2f} | 布林下轨:{boll_low:.2f}")
        except Exception as e:
            self.logger.error(f"❌ 风控计算失败: {str(e)[:100]}")
            self.trade_allowed = True

    def set_leverage(self):
        try:
            data = {
                "instId": self.symbol,
                "lever": str(self.leverage),
                "mgnMode": "cross",
                "posSide": "long",
            }
            result = self.client.request("POST", "/api/v5/account/set-leverage", data=data)
            if result:
                self.logger.info(f"✅ 杠杆设置成功 | {self.leverage}倍")
            else:
                self.logger.warning(f"⚠️  杠杆设置可能未生效")
        except Exception as e:
            self.logger.error(f"❌ 杠杆设置警告: {str(e)[:80]}")

    def check_position(self):
        try:
            params = {"instId": self.symbol}
            result = self.client.request("GET", "/api/v5/position/list", params=params)
            if result:
                long_pos = None
                for pos in result:
                    if pos['posSide'] == 'long' and pos['instId'] == self.symbol:
                        long_pos = pos
                        break
                if long_pos and float(long_pos['pos']) > 0:
                    self.position = 1
                    self.entry_price = float(long_pos['avgPx'])
                    self.logger.info(f"✅ 当前持仓 | 多头 {long_pos['pos']} 张 | 开仓价 {self.entry_price:.2f}")
                else:
                    self.position = 0
                    self.entry_price = None
                    self.logger.info("✅ 当前持仓 | 本地记录: 无持仓")
            else:
                status = "多头持仓" if self.position == 1 else "无持仓"
                self.logger.info(f"✅ 当前持仓 | 本地记录: {status}")
        except Exception as e:
            self.logger.error(f"ℹ️  持仓查询暂时失败: {str(e)[:50]}")

    def buy(self):
        if self.position != 0:
            self.logger.warning("⚠️  当前已有持仓，无法开多")
            return None
        # 计算下单张数
        order_size = self.calculate_order_size()
        if order_size <= 0:
            return None

        try:
            data = {
                "instId": self.symbol, "tdMode": "cross", "side": "buy",
                "ordType": "market", "sz": str(order_size), "posSide": "long"
            }
            result = self.client.request("POST", "/api/v5/trade/order", data=data)
            if result and len(result) > 0:
                self.position = 1
                # 核心修改：从订单结果中提取真实成交价格（而非last_price）
                self.entry_price = float(result[0].get('avgPx', self.last_price))
                self.logger.info(
                    f"🟢 开多成功 | 订单ID:{result[0]['ordId']} | 价格:{self.entry_price:.2f} | 张数:{order_size}")

                # 保存开仓记录
                trade_record = {
                    "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "type": "开多",
                    "price": self.entry_price,
                    "size": order_size,
                    "profit": 0.0,
                    "order_id": result[0]['ordId']
                }
                save_trade_record(trade_record)

                return result[0]
            else:
                self.logger.error("❌ 开多失败")
                return None
        except Exception as e:
            self.logger.error(f"❌ 开多失败: {str(e)[:100]}")
            return None

    def sell(self, is_force=False):
        if not is_force and self.position != 1:
            self.logger.warning("⚠️  当前无多头持仓，无法平仓")
            return None
        # 计算平仓张数
        order_size = self.calculate_order_size() if is_force else None
        if is_force and (order_size is None or order_size <= 0):
            self.logger.error("❌ 强制平仓失败：无法计算仓位")
            return None
        if self.last_price is None and not is_force:
            self.logger.warning("⚠️  无最新价格，无法下单")
            return None
        if is_force and self.last_price is None:
            self.logger.info("🔄 强制平仓：重新获取最新价格...")
            self.last_price = self.client.get_ticker_price(self.symbol)
            if self.last_price <= 0:
                self.logger.error("❌ 强制平仓失败：无法获取最新价格")
                return None

        try:
            data = {
                "instId": self.symbol, "tdMode": "cross", "side": "sell",
                "ordType": "market", "sz": str(order_size), "posSide": "long"
            }
            result = self.client.request("POST", "/api/v5/trade/order", data=data)
            if result and len(result) > 0:
                sell_price = float(result[0].get('avgPx', self.last_price))
                profit = (sell_price - self.entry_price) * order_size if self.entry_price else 0
                self.position = 0
                self.entry_price = None
                self.logger.info(
                    f"🔴 平仓成功 | 订单ID:{result[0]['ordId']} | 价格:{sell_price:.2f} | 盈亏:{profit:.2f} USDT")

                # 保存平仓记录
                trade_record = {
                    "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "type": "平仓",
                    "price": sell_price,
                    "size": order_size,
                    "profit": profit,
                    "order_id": result[0]['ordId']
                }
                save_trade_record(trade_record)

                return result[0]
            else:
                self.logger.error("❌ 平仓失败")
                return None
        except Exception as e:
            self.logger.error(f"❌ 平仓失败: {str(e)[:100]}")
            return None

    def force_close_position(self):
        self.logger.info("\n🔴 执行强制平仓操作...")
        max_retry = 3
        retry_count = 0
        while retry_count < max_retry and self.position == 1:
            self.logger.info(f"\n📌 平仓重试 {retry_count + 1}/{max_retry}")
            result = self.sell(is_force=True)
            if result:
                self.logger.info("✅ 强制平仓成功！")
                break
            retry_count += 1
            time.sleep(2)
        if self.position == 1:
            self.logger.error("❌ 强制平仓失败！请手动在OKX模拟盘平仓")
        else:
            self.logger.info("✅ 所有持仓已清空")

    def run_strategy(self):
        self.logger.info(f"{'=' * 60}")
        self.logger.info(f"📈 OKX合约模拟盘策略启动 | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"📊 交易对:{self.symbol} | 杠杆:{self.leverage}倍 | 仓位比例:{self.position_ratio * 100}%")
        self.logger.info(f"{'=' * 60}\n")

        # 初始化
        self.set_leverage()
        self.check_position()
        df_init = self.fetch_ohlcv('1h', 500)
        self.train(df_init)

        cycle = 1
        total_trades = 0

        try:
            while True:
                try:
                    self.logger.info(f"\n{'-' * 40} 第{cycle}轮循环 {'-' * 40}")
                    df = self.fetch_ohlcv('1h', 300)
                    self.boll_filter()
                    self.check_position()

                    # 定期重训模型
                    if cycle % 24 == 0 and not df.empty:
                        self.logger.info("🔄 定期重训模型...")
                        self.train(df)

                    # 生成交易信号
                    signal = self.signal(df)
                    self.logger.info(f"📊 交易信号:{'📈 开多' if signal == 1 else '📉 观望'} | 持仓:{self.position}")

                    # 执行交易
                    trade_executed = False
                    if self.trade_allowed:
                        if signal == 1 and self.position == 0:
                            # ========== 核心修改：开仓前执行SLTP概率检查 ==========
                            if self.check_pre_open_sltp(df):
                                result = self.buy()
                                trade_executed = result is not None
                            else:
                                self.logger.info("📉 因SLTP概率不达标，放弃开仓")

                    # 止盈止损检查
                    if self.position == 1 and not df.empty:
                        self.check_stop_loss_take_profit(df)

                    if trade_executed:
                        total_trades += 1

                    self.logger.info(f"\n📋 运行统计 | 总交易次数: {total_trades}")
                    cycle += 1
                    self.logger.info(f"⏳ 等待{self.cycle_interval}秒后继续...")
                    time.sleep(self.cycle_interval)

                except Exception as e:
                    self.logger.error(f"\n❌ 循环出错: {str(e)[:80]}")
                    time.sleep(30)

        except KeyboardInterrupt:
            self.logger.info("\n\n🛑 检测到用户终止程序！")
            if self.position == 1:
                self.force_close_position()
            else:
                self.logger.info("✅ 当前无持仓，无需平仓")
            self.logger.info(f"\n📊 策略运行结束 | 总交易次数: {total_trades}")
            self.logger.info(f"🕒 结束时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info(f"{'=' * 60}")


# ================= 策略启动入口 =================
if __name__ == "__main__":
    try:
        # 加载配置文件
        config = load_config("config.yaml")  # 按实际配置文件路径调整
        # 初始化策略
        trader = OKXSimFuturesTrader(config)
        # 启动策略
        trader.run_strategy()
    except Exception as e:
        print(f"❌ 策略启动失败: {str(e)}")
        # 强制平仓（如果有持仓）
        try:
            trader.force_close_position()
        except:
            pass