"""map/H/meta/config 로드·저장."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml


def load_config(path: str | Path) -> dict:
    """configs/*.yaml을 dict로 읽는다."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def load_map(lot_dir: str | Path) -> dict:
    """data/lots/<lot>/map.json을 읽는다."""
    return _read(Path(lot_dir) / "map.json")


def save_map(lot_dir: str | Path, map_data: dict) -> None:
    _write(Path(lot_dir) / "map.json", map_data)


def load_cam(cam_dir: str | Path) -> dict:
    """data/cams/<lot>_<cam>/H.json을 읽는다. 'H'는 np.ndarray(3,3)로 변환."""
    cam = _read(Path(cam_dir) / "H.json")
    cam["H"] = np.array(cam["H"], np.float64)
    return cam


def save_cam(cam_dir: str | Path, cam_data: dict) -> None:
    _write(Path(cam_dir) / "H.json", {**cam_data, "H": np.asarray(cam_data["H"]).tolist()})


def load_clip_meta(clip_dir: str | Path) -> dict:
    """data/clips/<clip_id>/meta.json을 읽는다."""
    return json.loads((Path(clip_dir) / "meta.json").read_text(encoding="utf-8"))


def read_frame(video: str | Path, frame_idx: int):
    """frame_idx번째 프레임 (처음부터 순서대로 세서 읽음, seek 금지 규칙)."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    for _ in range(frame_idx):
        if not cap.grab():
            raise ValueError(f"{video}: 프레임 {frame_idx}까지 읽지 못함")
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"{video}: 프레임 {frame_idx}를 읽지 못함")
    return frame
