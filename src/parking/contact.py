"""지면 접점 추출 (CLAUDE.md 5.4)."""
from __future__ import annotations

import numpy as np


def contact_point_img(mask: np.ndarray | None, bbox_img: tuple[float, float, float, float],
                      mode: str, image_size: tuple[int, int]) -> tuple[np.ndarray, bool]:
    """검출 1개의 지면 접점(영상 픽셀)과 truncated 여부를 돌려준다.

    mode: mask_bottom_center | bbox_bottom_center | bbox_center
    """
    raise NotImplementedError
