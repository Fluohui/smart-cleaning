import logging
from utils.path_tool import get_abs_path
import os

#日志保存的根目录
LOG_ROOT = get_abs_path("logs")

#确保日志的目录存在
os.makedirs(LOG_ROOT, exist_ok=True)

#日志格式配置   级别（levelname）；error  info  debug
DEFAULT_LOG_FORMAT = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

def get_logger(
        name: str = "agent",
        console_lever: int = logging.INFO,
        file_lever: int = logging.DEBUG,
        log_file = None
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    #避免重复添加Handler    防止重复打印
    if logger.handlers:
        return logger

    # 控制台Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_lever)
    console_handler.setFormatter(DEFAULT_LOG_FORMAT)

    logger.addHandler(console_handler)

    #文件Handler
    if log_file is None:
        from datetime import datetime
        log_file = os.path.join(LOG_ROOT, f"{name}_{datetime.now().strftime('%Y-%m-%d')}.log")
    else:
        log_file = os.path.join(LOG_ROOT, log_file)

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(file_lever)
    file_handler.setFormatter(DEFAULT_LOG_FORMAT)

    logger.addHandler(file_handler)

    return logger


#快捷获取日志器
logger = get_logger()