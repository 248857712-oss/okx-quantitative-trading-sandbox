import json
import os
from typing import Dict, Any

# 合约策略默认配置（更新为合约默认配置）
DEFAULT_CONFIG = {
    "okx": {
        "api_key": "",
        "api_secret": "",
        "api_passphrase": "",
        "symbol": "BTC-USDT-SWAP",
        "is_sim": True,  # 默认模拟盘
        "inst_type": "SWAP"  # 合约类型
    },
    "strategy": {
        "leverage": 10,
        "position_ratio": 0.1,  # 10%仓位
        "lr_weight": 0.6,
        "rf_weight": 0.4,
        "vote_threshold": 0.4,
        "tp_prob_threshold": 0.7,
        "sl_prob_threshold": 0.5,
        "cycle_interval": 60,  # 轮询间隔
        "boll_window": 20,  # 布林带窗口
        "boll_dev": 2  # 布林带标准差
    },
    "log": {
        "log_path": "./logs",
        "log_level": "INFO"
    },
    "proxy": {
        "http": "",
        "https": ""
    }
}


def load_config(config_path: str = "./config.json") -> Dict[str, Any]:
    """加载配置文件，并自动填充默认值+校验关键参数"""
    # 加载用户配置
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            user_config = json.load(f)
    else:
        # 无配置文件时生成默认配置
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONFIG, f, indent=4, ensure_ascii=False)
        print(f"⚠️ 配置文件不存在，已生成默认配置: {config_path}")
        return DEFAULT_CONFIG

    # 递归合并默认配置和用户配置（用户配置覆盖默认值）
    def merge_config(default: Dict, user: Dict) -> Dict:
        merged = default.copy()
        for k, v in user.items():
            if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
                merged[k] = merge_config(merged[k], v)
            else:
                merged[k] = v
        return merged

    final_config = merge_config(DEFAULT_CONFIG, user_config)

    # 关键参数校验
    required_okx_keys = ["api_key", "api_secret", "api_passphrase"]
    for key in required_okx_keys:
        if not final_config["okx"][key]:
            raise ValueError(f"❌ OKX配置缺失关键参数: {key}")

    # 策略参数范围校验
    if not (1 <= final_config["strategy"]["leverage"] <= 100):
        raise ValueError(f"❌ 杠杆倍数必须在1-100之间，当前值: {final_config['strategy']['leverage']}")
    if not (0 < final_config["strategy"]["position_ratio"] <= 1):
        raise ValueError(f"❌ 仓位比例必须在0-1之间，当前值: {final_config['strategy']['position_ratio']}")

    print("✅ 配置加载并校验成功")
    return final_config