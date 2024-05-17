import logging


def init_logger(logger_name, level="INFO"):
    logging.basicConfig(
        format="%(asctime)s:%(levelname)s: %(message)s",
        datefmt="%I:%M:%S:%p",
        level=logging.INFO,
    )
    logger = logging.getLogger(logger_name)
    if level != "INFO":
        logger.setLevel(logging.ERROR)
    return logger
