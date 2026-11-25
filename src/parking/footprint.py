"""(2차) 차 바닥 영역(footprint) 추정, 칸별 겹침 비율 (CLAUDE.md 5.5).

- 보이는 부분: 마스크 볼록 껍질의 아래 경계(바닥에 닿는 쪽)를 맵에 투영한 점들.
- 안 보이는 부분: 가장 가까운 칸 방향으로 정렬한 표준 차 크기 사각형으로 채운다.
  보이는 아래 경계 중 카메라에 가장 가까운 점이 차의 카메라 쪽 옆면·앞면이므로,
  칸의 폭(u)·길이(v) 축마다 그 끝에서 카메라 반대쪽으로 차 폭·길이만큼 편다.
- 겹침 비율 = (차 바닥 ∩ 칸) / 차 바닥 면적. 칸 면적(12.2m²)이 차(8.5m²)보다 커서
  칸 면적 기준이면 꽉 찬 차도 70%라, 논문 ③의 15%/85% 기준은 차 면적 기준이어야 쓸 수 있다.
"""
from __future__ import annotations

import cv2
import numpy as np
from shapely.geometry import Polygon

from parking.homography import transform_points
from parking.occupancy import ENCROACH, FREE, OCCUPIED


def bottom_chain_img(mask_polys_img: list) -> np.ndarray:
    """마스크 볼록 껍질의 아래쪽 사슬 (가장 왼쪽 점 ~ 가장 오른쪽 점, 영상 아래쪽 경계), (K,2)."""
    pts = np.concatenate([np.asarray(p, np.float32).reshape(-1, 2) for p in mask_polys_img])
    hull = cv2.convexHull(pts).reshape(-1, 2).astype(np.float64)
    x = hull[:, 0]
    l = hull[x == x.min()][hull[x == x.min()][:, 1].argmax()]  # 세로 변이면 아래쪽 끝
    r = hull[x == x.max()][hull[x == x.max()][:, 1].argmax()]
    if r[0] == l[0]:
        return hull
    line_y = l[1] + (r[1] - l[1]) * (hull[:, 0] - l[0]) / (r[0] - l[0])
    return hull[hull[:, 1] >= line_y - 1e-6]


def slot_axes(poly: Polygon) -> tuple[np.ndarray, np.ndarray]:
    """칸 폴리곤 [a, a+step, a+step+side, a+side] -> (u: 폭 방향, v: 길이 방향) 단위 벡터."""
    c = np.asarray(poly.exterior.coords)[:4]
    u, v = c[1] - c[0], c[3] - c[0]
    return u / np.linalg.norm(u), v / np.linalg.norm(v)


def _range(s: np.ndarray, away: float, size: float, center_below: float) -> tuple[float, float]:
    """한 축 위의 범위: 카메라 쪽 끝에서 카메라 반대쪽으로 size. 축이 시선과 거의 수직이면 가운데 기준."""
    if abs(away) < center_below:
        m = (s.min() + s.max()) / 2
        return m - size / 2, m + size / 2
    near = s.min() if away > 0 else s.max()
    return (near, near + size) if away > 0 else (near - size, near)


def estimate_footprint(chain_map: np.ndarray, away_map: np.ndarray, u: np.ndarray, v: np.ndarray,
                       car_size: tuple[float, float], center_below: float = 0.25) -> Polygon:
    """보이는 아래 경계(맵, (K,2)) + 카메라 반대 방향(단위 벡터) + 칸 축 -> 차 바닥 사각형 (맵 좌표).

    car_size = (폭, 길이). 폭은 u(칸 폭) 축, 길이는 v(칸 길이) 축에 놓는다.
    """
    chain_map = np.asarray(chain_map, np.float64).reshape(-1, 2)
    u0, u1 = _range(chain_map @ u, float(away_map @ u), car_size[0], center_below)
    v0, v1 = _range(chain_map @ v, float(away_map @ v), car_size[1], center_below)
    return Polygon([u0 * u + v0 * v, u1 * u + v0 * v, u1 * u + v1 * v, u0 * u + v1 * v])


def looks_crosswise(chain_map: np.ndarray, u: np.ndarray, v: np.ndarray, min_len: float, ratio: float) -> bool:
    """보이는 바닥 경계가 칸 폭 방향(u)으로 min_len 이상, 칸 길이 방향(v)의 ratio배 이상 길면 칸과 수직으로 선 차."""
    chain_map = np.asarray(chain_map, np.float64).reshape(-1, 2)
    su, sv = np.ptp(chain_map @ u), np.ptp(chain_map @ v)
    return su >= min_len and su >= ratio * sv


def footprint_from_det(mask_polys_img: list, H: np.ndarray, away_map: np.ndarray, slot_poly: Polygon,
                       car_size: tuple[float, float], crosswise: tuple[float, float] | None = None) -> Polygon:
    """검출 1개 -> 차 바닥 사각형 (맵). 차 방향은 slot_poly(가장 가까운 칸) 방향으로 가정.

    crosswise=(min_len, ratio)가 있고 보이는 바닥 경계가 칸과 수직으로 길면 사각형을 눕힌다 (통로를 가로지르는 차).
    주차된 차는 바닥 경계가 높이 때문에 옆으로 퍼져 보이기도 해서, 호출 쪽에서 통로 위 차에만 켠다.
    """
    chain_map = transform_points(bottom_chain_img(mask_polys_img).astype(np.float32), H)
    u, v = slot_axes(slot_poly)
    if crosswise and looks_crosswise(chain_map, u, v, *crosswise):
        car_size = car_size[::-1]
    return estimate_footprint(chain_map, away_map, u, v, car_size)


def slot_overlaps(fp: Polygon, slots: dict[str, Polygon]) -> dict[str, float]:
    """칸별 겹침 비율 (차 바닥 면적 기준). 0인 칸은 뺀다."""
    out = {}
    for k, p in slots.items():
        a = fp.intersection(p).area
        if a > 0:
            out[k] = a / fp.area
    return out


def footprint_states(overlaps: list[dict[str, float]], slots: dict, tau_occupy: float,
                     tau_straddle: float) -> dict[str, int]:
    """차마다 (겹침 비율 = 차 바닥 면적 기준):
    - tau_occupy 이상 겹친 칸은 모두 점유(1). 선을 밟은 차는 걸친 칸도 막으므로 빈칸으로 안내하지 않는다.
    - 가장 많이 겹친 칸 말고도 tau_straddle 이상 겹친 칸이 있으면, 그 칸들과 가장 많이 겹친 칸 모두 침범(2).
    칸에 침범이 하나라도 있으면 2 (점유보다 우선).
    """
    states = dict.fromkeys(slots, FREE)
    for ov in overlaps:
        hit = [k for k, r in ov.items() if r >= tau_occupy]
        if not hit:
            continue
        best = max(ov, key=ov.get)
        other = [k for k in hit if k != best and ov[k] >= tau_straddle]
        for k in hit:
            states[k] = max(states[k], ENCROACH if (other and k in (best, *other)) else OCCUPIED)
    return states
