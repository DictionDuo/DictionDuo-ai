import logging
import os

def setup_logger(save_path: str, name: str = "train"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # 중복 방지

    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # 콘솔 출력 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 파일 출력 핸들러
    file_handler = logging.FileHandler(save_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger