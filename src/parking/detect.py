"""YOLO-seg 추론, 결과 JSONL 캐시."""
from __future__ import annotations

from pathlib import Path


def run_detection(video_path: str | Path, cfg: dict, cache_path: str | Path) -> Path:
    """원본 프레임을 sampling.fps로 샘플링해 검출하고 프레임별 JSONL로 저장한다."""
    raise NotImplementedError


def load_cache(cache_path: str | Path) -> list[dict]:
    raise NotImplementedError
