import logging


def get_logger(name: str, log_file: str) -> logging.Logger:
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter("%(message)s")
    file_handler.setFormatter(file_format)

    stream_handler = logging.StreamHandler()
    # stream_handler.setLevel(logging.DEBUG)
    stream_handler.setLevel(logging.INFO)
    stream_format = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s")
    stream_handler.setFormatter(stream_format)

    logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, stream_handler])

    logger = logging.getLogger(name)
    return logger
