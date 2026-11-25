"""(비교용) 칸 3D 상자의 안 가려진 부분으로 점유 판정 (Masmoudi 외, "Parking Spaces Modelling for Inter Spaces
Occlusion Handling"의 아이디어: 칸을 3D 상자로 보고, 앞 칸 상자에 가리지 않는 중요한 부분만 쓴다).

- 칸 상자 = 칸 바닥 폴리곤 x 높이 h. 8꼭짓점 투영의 볼록 껍질 = 영상에서 그 칸에 선 차가 차지할 수 있는 영역.
- 안 가려진 부분 = 칸 상자 영역 - (카메라에 더 가까운 칸들의 상자 영역 합집합).
- 점유 = 안 가려진 부분 중 차 마스크로 덮인 비율 >= tau.
본 논문은 상자 안 특징으로 분류기를 쓰지만, 여기서는 YOLO 마스크 겹침 비율로 대신한다.
"""
from __future__ import annotations

import numpy as np
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import unary_union

from parking.box3d import project
from parking.occupancy import FREE, OCCUPIED


def _mask_shape(mask_polys_img: list):
    return unary_union([Polygon(p).buffer(0) for p in mask_polys_img if len(p) > 2])


def camera_center(P: np.ndarray) -> np.ndarray:
    """투영 행렬의 카메라 위치 (맵, (3,))."""
    return -np.linalg.inv(P[:, :3]) @ P[:, 3]


def visible_regions(slots: dict[str, Polygon], P: np.ndarray, height: float) -> dict[str, Polygon]:
    """칸별 안 가려진 상자 영역 (영상 픽셀). 칸 중심이 카메라에 가까운 순서로 앞 칸들의 영역을 뺀다."""
    C = camera_center(P)[:2]
    hulls = {}
    for k, p in slots.items():
        base = np.asarray(p.exterior.coords)[:-1]
        pts3 = np.vstack([np.c_[base, np.zeros(len(base))], np.c_[base, np.full(len(base), height)]])
        hulls[k] = MultiPoint(project(P, pts3)).convex_hull
    order = sorted(slots, key=lambda k: np.linalg.norm(np.asarray(slots[k].centroid.coords[0]) - C))
    out, front = {}, Polygon()
    for k in order:
        out[k] = hulls[k].difference(front)
        front = unary_union([front, hulls[k]])
    return out


def volume_states(mask_polys: list, regions: dict[str, Polygon], tau: float) -> tuple[dict[str, int], dict[str, float]]:
    """검출 마스크들 -> (칸 상태 0/1, 칸별 덮인 비율)."""
    masks = unary_union([_mask_shape(m) for m in mask_polys if m]) if mask_polys else Polygon()
    cov = {k: (r.intersection(masks).area / r.area if r.area > 0 else 0.0) for k, r in regions.items()}
    return {k: OCCUPIED if c >= tau else FREE for k, c in cov.items()}, cov
