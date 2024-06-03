import functools
import logging


@functools.cache
def log_once(logger: logging.Logger, message: str, level: int):
    logger.log(level=level, msg=message)
