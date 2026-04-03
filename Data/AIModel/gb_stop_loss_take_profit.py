import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("OKXQuantStrategy")


class GBSLTPModel:
    """止盈止损模型（修复过拟合+成本差特征+概率平滑）"""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        # 模型参数：增加正则化，防止过拟合
        self.tp_model = GradientBoostingClassifier(
            n_estimators=80,  # 减少弱学习器数量
            max_depth=3,  # 降低树深度
            learning_rate=0.1,  # 提高学习率
            subsample=0.8,
            max_features='sqrt',  # 随机选特征，防止过拟合
            random_state=random_state
        )
        self.sl_model = GradientBoostingClassifier(
            n_estimators=80,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            max_features='sqrt',
            random_state=random_state
        )
        self.tp_trained = False
        self.sl_trained = False
        self.feature_cols = ['ret', 'range', 'vol_ratio', 'ma_ratio', 'bb_pos', 'rsi', 'price_diff_ratio']
        self.original_cols = ['open', 'high', 'low', 'close', 'vol']

    def extract_features(self, df, entry_price=None):
        if df.empty:
            logger.warning("❌ 特征提取失败：输入数据为空")
            return pd.DataFrame()

        missing_cols = [col for col in self.original_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"❌ 缺失列 {missing_cols}")
            return pd.DataFrame()

        df = df.copy()

        # 基础特征
        df['ret'] = (df['close'] - df['open']) / df['open'].replace(0, np.nan).fillna(0.0)
        df['range'] = (df['high'] - df['low']) / df['open'].replace(0, np.nan).fillna(0.0)

        # 成交量比率
        df['vol_ratio'] = df['vol'] / df['vol'].rolling(20, min_periods=1).mean().replace(0, 1.0).fillna(1.0)

        # 均线特征
        df['ma5'] = df['close'].rolling(5, min_periods=1).mean()
        df['ma20'] = df['close'].rolling(20, min_periods=1).mean()
        df['ma_ratio'] = (df['ma5'] / df['ma20'].replace(0, 1.0)).fillna(1.0)

        # 布林带
        df['bb_mid'] = df['close'].rolling(20, min_periods=1).mean()
        df['bb_std'] = df['close'].rolling(20, min_periods=1).std().fillna(0.01)
        df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
        bb_range = df['bb_upper'] - df['bb_lower']
        df['bb_pos'] = ((df['close'] - df['bb_lower']) / bb_range.replace(0, 0.01)).clip(0, 1)

        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
        loss = -delta.where(delta < 0, 0).rolling(14, min_periods=1).mean()
        rs = gain / loss.replace(0, 0.01)
        df['rsi'] = (100 - (100 / (1 + rs))).fillna(50.0).clip(0, 100)

        # 成本差特征（关键！）
        if entry_price and entry_price > 0:
            df['price_diff_ratio'] = (df['close'] - entry_price) / entry_price
            logger.info(
                f"📊 成本差特征生效 | 开仓价:{entry_price} | 最新价:{df['close'].iloc[-1]} | 差值比:{df['price_diff_ratio'].iloc[-1]:.4f}")
        else:
            df['price_diff_ratio'] = 0.0
            logger.warning("⚠️ 未传入开仓价，成本差特征为0")

        df = df.dropna(subset=self.feature_cols, how='all')
        logger.info(f"✅ 特征提取完成 | 有效样本数:{len(df)}")
        return df

    def create_labels(self, df, tp_threshold=0.002, sl_threshold=0.002):
        if df.empty:
            logger.warning("❌ 标签创建失败：数据为空")
            return pd.DataFrame()

        if 'close' not in df.columns:
            logger.error("❌ 无close列")
            return pd.DataFrame()

        df = df.copy()

        # 预测未来2根K线（平衡敏感度和稳定性）
        df['future_ret'] = df['close'].shift(-1) / df['close'] - 1
        df['future_ret'] = df['future_ret'].fillna(0.0)

        df['tp_label'] = (df['future_ret'] >= tp_threshold).astype(int)
        df['sl_label'] = (df['future_ret'] <= -sl_threshold).astype(int)

        if len(df) > 2:
            df = df.iloc[:-2]

        tp_ratio = df['tp_label'].mean()
        sl_ratio = df['sl_label'].mean()
        neutral_ratio = 1 - tp_ratio - sl_ratio

        logger.info(f"✅ 标签创建 | 止盈:{tp_ratio:.2%} | 止损:{sl_ratio:.2%} | 中性:{neutral_ratio:.2%}")
        return df

    def train(self, df, tp_threshold=0.01, sl_threshold=0.01):
        df_features = self.extract_features(df)
        if len(df_features) < 50:
            logger.error(f"❌ 样本数不足 {len(df_features)}")
            return

        df_labeled = self.create_labels(df_features, tp_threshold, sl_threshold)
        if len(df_labeled) < 20:
            logger.error(f"❌ 带标签样本数不足 {len(df_labeled)}")
            return

        X = df_labeled[self.feature_cols]
        y_tp = df_labeled['tp_label']
        y_sl = df_labeled['sl_label']

        # 划分训练/测试集（验证泛化能力）
        X_train, X_test, y_tp_train, y_tp_test = train_test_split(
            X, y_tp, test_size=0.2, random_state=self.random_state, stratify=y_tp
        )
        X_train_sl, X_test_sl, y_sl_train, y_sl_test = train_test_split(
            X, y_sl, test_size=0.2, random_state=self.random_state, stratify=y_sl
        )

        # 过采样（仅训练集）
        ros = RandomOverSampler(random_state=self.random_state)
        X_train_resampled, y_tp_resampled = ros.fit_resample(X_train, y_tp_train)
        X_train_sl_resampled, y_sl_resampled = ros.fit_resample(X_train_sl, y_sl_train)

        # 标准化（仅用训练集拟合）
        self.scaler.fit(X_train_resampled)
        X_train_scaled = self.scaler.transform(X_train_resampled)
        X_test_scaled = self.scaler.transform(X_test)
        X_train_sl_scaled = self.scaler.transform(X_train_sl_resampled)
        X_test_sl_scaled = self.scaler.transform(X_test_sl)

        # 训练模型
        self.tp_model.fit(X_train_scaled, y_tp_resampled)
        self.sl_model.fit(X_train_sl_scaled, y_sl_resampled)

        # 验证泛化能力（关键！不再用训练集验证）
        tp_train_acc = accuracy_score(y_tp_resampled, self.tp_model.predict(X_train_scaled))
        tp_test_acc = accuracy_score(y_tp_test, self.tp_model.predict(X_test_scaled))
        sl_train_acc = accuracy_score(y_sl_resampled, self.sl_model.predict(X_train_sl_scaled))
        sl_test_acc = accuracy_score(y_sl_test, self.sl_model.predict(X_test_sl_scaled))

        self.tp_trained = True
        self.sl_trained = True

        logger.info(f"✅ 模型训练完成 | 止盈训练准确率:{tp_train_acc:.2%} | 止盈测试准确率:{tp_test_acc:.2%}")
        logger.info(f"✅ 模型训练完成 | 止损训练准确率:{sl_train_acc:.2%} | 止损测试准确率:{sl_test_acc:.2%}")

    def predict(self, df, entry_price=None, debug=False):
        if not self.tp_trained or not self.sl_trained:
            logger.error("❌ 模型未训练")
            return 0.0, 0.0

        if df.empty:
            logger.error("❌ 输入数据为空")
            return 0.0, 0.0

        # 提取特征（必须传入开仓价）
        df_features = self.extract_features(df, entry_price=entry_price)
        if df_features.empty:
            logger.error("❌ 特征提取后为空")
            return 0.0, 0.0

        latest_features = df_features.iloc[-1][self.feature_cols]
        latest_df = latest_features.to_frame().T

        if debug:
            logger.info(f"🔍 最新特征值：{latest_features.to_dict()}")

        try:
            latest_scaled = self.scaler.transform(latest_df)
            tp_prob = self.tp_model.predict_proba(latest_scaled)[0][1]
            sl_prob = self.sl_model.predict_proba(latest_scaled)[0][1]

            # 概率平滑（避免极端值）
            tp_prob = np.clip(tp_prob, 0.05, 0.95)
            sl_prob = np.clip(sl_prob, 0.05, 0.95)

            logger.info(f"✅ 预测完成 | 止盈概率:{tp_prob:.2%} | 止损概率:{sl_prob:.2%}")
            return tp_prob, sl_prob
        except Exception as e:
            logger.error(f"❌ 预测异常：{str(e)}", exc_info=True)
            return 0.0, 0.0

    def reset(self):
        self.tp_trained = False
        self.sl_trained = False
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        logger.info("✅ 模型已重置")


# 测试代码
if __name__ == "__main__":
    np.random.seed(42)
    dates = pd.date_range(start='2026-01-01', periods=300, freq='1h')
    close = 95000 + np.cumsum(np.random.randn(300) * 100)
    test_df = pd.DataFrame({
        'open': close + np.random.randn(300) * 50,
        'high': close + np.random.randn(300) * 100,
        'low': close - np.random.randn(300) * 100,
        'close': close,
        'vol': np.random.randint(1000, 10000, 300)
    }, index=dates)

    model = GBSLTPModel()
    model.train(test_df)
    tp_prob, sl_prob = model.predict(test_df, entry_price=95000, debug=True)
    print(f"\n止盈概率：{tp_prob:.2%} | 止损概率：{sl_prob:.2%}")