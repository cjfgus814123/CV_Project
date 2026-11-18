"""1차 점유 판정, 가시성, 시간 보정 (CLAUDE.md 5.5)."""
from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon

FREE, OCCUPIED, ENCROACH, UNKNOWN = 0, 1, 2, -1


def assign_slots(contacts_map: np.ndarray, slots: dict[str, Polygon], tol_m: float) -> dict[str, int]:
    """접점(맵 좌표)이 들어간 칸 → 1, 나머지 → 0. 경계 근처는 tol_m 안의 가장 가까운 칸."""
    raise NotImplementedError


def slot_visibility(slots: dict[str, Polygon], visible_map: Polygon) -> dict[str, float]:
    """칸별 가시 비율 (칸 면적 중 가시 영역과 겹치는 비율)."""
    raise NotImplementedError


def apply_visibility(states: dict[str, int], vis: dict[str, float], vis_unknown: float) -> dict[str, int]:
    """가시 비율 < vis_unknown인 칸을 -1로 바꾼다."""
    raise NotImplementedError


def temporal_smooth(history: list[dict[str, int]], n: int) -> dict[str, int]:
    """최근 n프레임 다수결로 칸 상태를 정한다."""
    raise NotImplementedError
