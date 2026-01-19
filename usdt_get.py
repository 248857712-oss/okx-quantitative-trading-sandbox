import ccxt
import json
import requests

# 禁用SSL警告（代理环境下避免证书报错）
requests.packages.urllib3.disable_warnings()

# ========== 核心配置 ==========
# 代理配置（根据实际情况修改协议和端口）
PROXY_URL = "http://127.0.0.1:10808"  # SOCKS5请改为 socks5://127.0.0.1:10808

# OKX模拟盘API配置
API_CONFIG = {
    "apiKey": "2661e031-b477-4f5d-8e2f-eb6f37b4c843",
    "secret": "DFB6E3A73B1D6A6939EA8A25B0AE7AE7",
    "password": "!Lxz20071125",  # OKX的passphrase对应ccxt的password
    "enableRateLimit": True,  # 启用速率限制，避免请求超限
    "options": {
        "defaultType": "spot",  # 默认查询现货资产
        "fetchBalance": "all"  # 获取所有类型资产
    }
}


# 配置带代理的requests session（核心修复）
def create_proxy_session():
    """创建带代理的requests会话"""
    session = requests.Session()
    session.proxies = {
        "http": PROXY_URL,
        "https": PROXY_URL,
    }
    # 增加超时设置，避免卡死
    session.timeout = 15
    return session


def test_proxy_connectivity():
    """测试代理能否访问OKX官网"""
    try:
        session = create_proxy_session()
        response = session.get("https://www.okx.com")
        if response.status_code == 200:
            print("✅ 代理连通性测试成功")
            return True
        else:
            print(f"❌ 代理测试失败，状态码: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 代理连接失败: {str(e)}")
        return False


def get_okx_sandbox_balance():
    """获取OKX模拟盘账户资产（核心功能）"""
    try:
        # 1. 创建带代理的session
        proxy_session = create_proxy_session()

        # 2. 初始化OKX模拟盘实例（关键：传入自定义session）
        okx = ccxt.okx({
            **API_CONFIG,
            'sandbox': True,  # 启用模拟盘模式
            'hostname': 'www.okx.com',  # 模拟盘域名
            'session': proxy_session  # 传入配置好代理的session（核心修复）
        })

        # 3. 加载市场数据（避免缓存问题）
        okx.load_markets()

        # 4. 获取账户资产
        print("🔄 正在获取OKX模拟盘账户资产...")
        balance = okx.fetch_balance()

        # 5. 解析并打印资产数据
        print("\n=== OKX模拟盘账户资产汇总 ===")
        # 打印总资产（USDT计价）
        total_usdt = balance.get('total', {}).get('USDT', 0)
        print(f"总资产(USDT): {total_usdt}")

        # 提取非零资产明细
        non_zero_assets = {}
        for coin, amount in balance['total'].items():
            if float(amount) > 0.000001:  # 过滤极小数值
                non_zero_assets[coin] = {
                    '总计': round(float(amount), 6),
                    '可用': round(float(balance['free'].get(coin, 0)), 6),
                    '已用': round(float(balance['used'].get(coin, 0)), 6)
                }

        # 打印非零资产
        if non_zero_assets:
            print("\n=== 非零资产明细 ===")
            for coin, info in non_zero_assets.items():
                print(f"{coin}: {info}")
        else:
            print("\n⚠️  模拟盘账户暂无可用资产")

        return balance

    # 针对性异常处理
    except ccxt.AuthenticationError as e:
        print(f"\n❌ 认证失败: {str(e)}")
        print("排查方向：")
        print("1. API Key/Secret/Passphrase 是否输入正确")
        print("2. 确认是模拟盘API（不是实盘API）")
        print("3. 检查API的IP白名单配置（若开启）")
        return None
    except ccxt.NetworkError as e:
        print(f"\n❌ 网络错误: {str(e)}")
        print("排查方向：")
        print("1. 代理地址/协议/端口是否正确")
        print("2. 代理软件是否正常运行")
        print("3. 尝试切换代理协议（http ↔ socks5）")
        return None
    except Exception as e:
        print(f"\n❌ 获取资产失败: {str(e)}")
        return None


# 主执行逻辑
if __name__ == "__main__":
    # 先测试代理连通性
    if test_proxy_connectivity():
        # 再获取模拟盘资产
        get_okx_sandbox_balance()