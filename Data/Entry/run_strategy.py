from main import  OKXFuturesTrader
from config_utils import load_config

if __name__ == '__main__':
    try:
        # 加载配置
        config = load_config()
        # 初始化策略
        trader = OKXFuturesTrader(config)
        # 运行策略
        trader.run_strategy()
    except Exception as e:
        print(f"❌ 策略启动失败: {e}")
        # 如果日志已经初始化，可以用日志输出，否则用print
        import traceback
        traceback.print_exc()