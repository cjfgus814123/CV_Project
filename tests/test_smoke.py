"""뼈대 확인: 모든 모듈 import, default.yaml 필수 키."""
import importlib
from pathlib import Path

import pytest

from parking.io import load_config

ROOT = Path(__file__).resolve().parents[1]
MODULES = ["io", "calibrate", "homography", "map_builder", "detect", "contact", "track",
           "occupancy", "footprint", "illegal", "baseline", "render", "pipeline"]


@pytest.mark.parametrize("name", MODULES)
def test_import(name):
    importlib.import_module(f"parking.{name}")


def test_default_config():
    cfg = load_config(ROOT / "configs/default.yaml")
    for key in ["model", "sampling", "geometry", "contact", "occupancy", "footprint", "illegal"]:
        assert key in cfg, key
    assert cfg["contact"]["mode"] in {"mask_bottom_center", "bbox_bottom_center", "bbox_center"}
    assert cfg["occupancy"]["vis_unknown"] < cfg["occupancy"]["vis_judge"]
