import logging

logger_format = "%(levelname)s %(filename)s:%(funcName)s:%(lineno)d %(message)s"
logging.basicConfig(level=logging.WARNING, format=logger_format)

LEVEL = logging.INFO
logger = logging.getLogger()
logger.setLevel(LEVEL)
