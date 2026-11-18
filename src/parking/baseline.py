"""논문 ② 방식 베이스라인: 영상 위 칸 폴리곤 + 박스 중심점 판정."""
from __future__ import annotations

import numpy as np


def baseline_occupancy(bboxes_img: np.ndarray, slots_img: dict[str, np.ndarray]) -> dict[str, int]:
    """박스 중심이 영상 칸 폴리곤(map 칸을 H^-1로 투영) 안에 있으면 점유."""
    raise NotImplementedError
