"""1차 점유 판정, 가시성, 시간 보정 (CLAUDE.md 5.5). 모든 좌표는 맵 좌표(m)."""
from __future__ import annotations

from collections import Counter

import numpy as np
from shapely.geometry import Point, Polygon

FREE, OCCUPIED, ENCROACH, UNKNOWN = 0, 1, 2, -1
# 다수결 동점일 때 우선순위: 점유 쪽으로 (빈칸 Precision 우선 = "차 있는 칸을 빈칸으로 안내" 방지)
_TIE_PRIORITY = {OCCUPIED: 0, ENCROACH: 1, FREE: 2, UNKNOWN: 3}


def slots_from_map(map_data: dict) -> dict[str, Polygon]:
    """map.json의 slots -> {slot_id: Polygon}"""
    return {s["id"]: Polygon(s["polygon"]) for s in map_data["slots"]}


def nearest_slot(pt_map, slots: dict[str, Polygon], tol_m: float) -> str | None:
    """점이 들어간 칸. 칸 밖이면 tol_m 안의 가장 가까운 칸, 없으면 None (통로 위 차 등).

    여러 칸에 동시에 걸리면(경계 위) 칸 중심이 더 가까운 칸.
    """
    p = Point(float(pt_map[0]), float(pt_map[1]))
    best = min(slots, key=lambda k: (slots[k].distance(p), slots[k].centroid.distance(p)))
    return best if slots[best].distance(p) <= tol_m else None


def assign_slots(contacts_map: np.ndarray, slots: dict[str, Polygon], tol_m: float) -> dict[str, int]:
    """접점(맵 좌표, (N,2))이 배정된 칸 -> 1, 나머지 -> 0."""
    states = dict.fromkeys(slots, FREE)
    for pt in np.asarray(contacts_map, dtype=np.float64).reshape(-1, 2):
        sid = nearest_slot(pt, slots, tol_m)
        if sid is not None:
            states[sid] = OCCUPIED
    return states


def slot_visibility(slots: dict[str, Polygon], visible_map: Polygon) -> dict[str, float]:
    """칸별 가시 비율 = (칸 ∩ 가시 영역) 면적 / 칸 면적."""
    return {k: (p.intersection(visible_map).area / p.area if p.area > 0 else 0.0) for k, p in slots.items()}


def apply_visibility(states: dict[str, int], vis: dict[str, float], vis_unknown: float) -> dict[str, int]:
    """가시 비율 < vis_unknown인 칸 -> -1 (판단불가). vis_unknown~vis_judge(불확실)는 그대로 두고 render가 표시."""
    return {k: (UNKNOWN if vis.get(k, 0.0) < vis_unknown else s) for k, s in states.items()}


def temporal_smooth(history: list[dict[str, int]], n: int) -> dict[str, int]:
    """최근 n프레임 다수결. 동점이면 점유 쪽. 기록이 n개보다 적으면 있는 만큼만 쓴다."""
    recent = history[-n:]
    out = {}
    for k in recent[-1]:
        votes = Counter(h[k] for h in recent if k in h)
        out[k] = min(votes, key=lambda s: (-votes[s], _TIE_PRIORITY.get(s, 9)))
    return out
