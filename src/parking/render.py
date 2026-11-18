"""Split-View 영상, 2D 맵 그리기."""
from __future__ import annotations

import numpy as np


def draw_frame_view(frame_img: np.ndarray, dets: list[dict], slots_img: dict, states: dict[str, int]) -> np.ndarray:
    """왼쪽: 원본 + 칸 폴리곤(H^-1) + 검출."""
    raise NotImplementedError


def draw_map_view(bev_img: np.ndarray, map_data: dict, states: dict[str, int]) -> np.ndarray:
    """오른쪽: map_bev 위 칸 상태 색칠 + 빈자리 수."""
    raise NotImplementedError


def split_view(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    raise NotImplementedError
