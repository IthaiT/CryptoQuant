import sys
from loguru import logger

def _setup_logger(log_type: str = "simple") -> None:
    """配置logger格式"""
    logger.remove()
    if log_type == "simple":
        logger.add(
            sys.stdout,
            format="| <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="DEBUG"
        )
    elif log_type == "verbose":
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="DEBUG"
        )

# 初始化logger为简单模式
_setup_logger("simple")
