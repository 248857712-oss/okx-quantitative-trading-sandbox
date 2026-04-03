import logging
# 关键修复：显式导入 handlers 子模块
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime


def init_logger(log_path, log_level=logging.INFO):
    # 创建日志目录
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # 按日期分割日志文件（避免单个文件过大）
    log_filename = os.path.join(log_path, f"okx_spot_strategy_{datetime.now().strftime('%Y%m%d')}.log")

    # 定义日志格式：时间 + 级别 + 模块 + 消息 + 关键标签
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # 文件处理器（按大小轮转，备份5个文件）
    # 修复：直接使用导入的 RotatingFileHandler，而非 logging.handlers.RotatingFileHandler
    file_handler = RotatingFileHandler(
        log_filename,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)

    # 初始化logger
    logger = logging.getLogger("OKXSpotQuantStrategy")
    logger.setLevel(log_level)
    logger.handlers.clear()  # 清除重复处理器，解决日志重复问题
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# 新增：关键操作日志装饰器（统一记录交易动作）
def trade_logger(func):
    def wrapper(*args, **kwargs):
        logger = logging.getLogger("OKXSpotQuantStrategy")
        func_name = func.__name__
        logger.info(f"📌 开始执行: {func_name}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"✅ 执行成功: {func_name}")
            return result
        except Exception as e:
            logger.error(f"❌ 执行失败: {func_name} | 错误: {str(e)[:100]}", exc_info=True)
            raise

    return wrapper