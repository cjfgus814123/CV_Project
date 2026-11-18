"""map/H/meta/config 로드·저장."""
from __future__ import annotations

import json
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """configs/*.yaml을 dict로 읽는다."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_map(lot_dir: str | Path) -> dict:
    """data/lots/<lot>/map.json을 읽는다."""
    raise NotImplementedError


def save_map(lot_dir: str | Path, map_data: dict) -> None:
    raise NotImplementedError


def load_cam(cam_dir: str | Path) -> dict:
    """data/cams/<lot>_<cam>/H.json을 읽는다. 'H'는 np.ndarray(3,3)로 변환."""
    raise NotImplementedError


def save_cam(cam_dir: str | Path, cam_data: dict) -> None:
    raise NotImplementedError


def load_clip_meta(clip_dir: str | Path) -> dict:
    """data/clips/<clip_id>/meta.json을 읽는다."""
    return json.loads((Path(clip_dir) / "meta.json").read_text(encoding="utf-8"))
