import os
import json
from glob import glob
from collections import Counter

def count_error_tags(json_dir):
    """
    error_tags가 있는 JSON만 대상으로 오류 종류를 세어 Counter 반환
    """
    counter = Counter()
    json_paths = glob(os.path.join(json_dir, "*.json"))

    for path in json_paths:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            if not (
                "RecordingMetadata" in data and
                "phonemic" in data["RecordingMetadata"] and
                "error_tags" in data["RecordingMetadata"]["phonemic"]
            ):
                continue

            for e in data["RecordingMetadata"]["phonemic"]["error_tags"]:
                tag = e.get("error_tag")
                if tag:
                    counter[tag] += 1

        except Exception as e:
            print(f"[Error reading {path}]: {e}")
    return counter


def build_error_map(json_dir, save_path=None):
    """
    오류 태그 빈도로부터 error_class_map을 생성하고 선택적으로 저장
    """
    counts = count_error_tags(json_dir)
    error_map = {"정상": 0}
    for idx, (tag, _) in enumerate(sorted(counts.items()), start=1):
        error_map[tag] = idx

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(error_map, f, indent=2, ensure_ascii=False)
        print(f"[Saved] error_class_map → {save_path}")

    return error_map