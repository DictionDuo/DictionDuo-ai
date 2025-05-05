import json
import argparse

def convert_config_to_namespace(config_path: str) -> argparse.Namespace:
    """
    JSON 파일을 argparse.Namespace 객체로 변환

    Args:
        config_path (str): JSON 구성 파일 경로

    Returns:
        argparse.Namespace: config dict를 네임스페이스 객체로 변환한 결과
    """
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    return argparse.Namespace(**config_dict)