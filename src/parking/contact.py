"""지면 접점 추출 (CLAUDE.md 5.4).

H는 바닥 평면 위의 점만 정확히 옮긴다. 그래서 차의 높이가 있는 점(박스 중심, 지붕)이 아니라
마스크가 바닥에 닿는 아래 경계에서 점 하나를 고른다.
"""
from __future__ import annotations

import cv2
import numpy as np

from parking.homography import transform_points

MODES = ("mask_bottom_center", "bbox_bottom_center", "bbox_center")


def hull_bottom_center(mask_polys_img: list) -> np.ndarray:
    """마스크(다각형 목록)의 볼록 껍질에서, 가로 중앙 x의 가장 아래 y.

    볼록 껍질을 쓰는 이유: 가로등·나무가 마스크 아래쪽을 파고들거나 둘로 갈라도
    껍질이 그 홈을 메워, 접점이 홈 위로 튀어 뒤 칸에 찍히지 않는다.
    """
    pts = np.concatenate([np.asarray(p, np.float32).reshape(-1, 2) for p in mask_polys_img])
    hull = cv2.convexHull(pts).reshape(-1, 2).astype(np.float64)
    xc = (hull[:, 0].min() + hull[:, 0].max()) / 2
    ys = []
    for p, q in zip(hull, np.roll(hull, -1, axis=0)):
        lo, hi = sorted((p[0], q[0]))
        if lo <= xc <= hi:
            ys.append(p[1] if hi == lo else p[1] + (q[1] - p[1]) * (xc - p[0]) / (q[0] - p[0]))
    return np.array([xc, max(ys)], np.float32)


def is_truncated(bbox_img, image_size: tuple[int, int], edge_margin_px: float) -> bool:
    """박스가 왼쪽·오른쪽·아래 경계에 닿으면 잘린 검출. 위쪽은 바닥 접점과 무관해서 제외."""
    x1, _, x2, y2 = bbox_img
    w, h = image_size
    return x1 <= edge_margin_px or x2 >= w - 1 - edge_margin_px or y2 >= h - 1 - edge_margin_px


def contact_point_img(mask_polys_img: list | None, bbox_img, mode: str,
                      image_size: tuple[int, int], edge_margin_px: float) -> tuple[np.ndarray, bool]:
    """검출 1개의 지면 접점(영상 픽셀, (2,) float32)과 truncated 여부.

    mode: mask_bottom_center | bbox_bottom_center | bbox_center.
    마스크가 없으면 mask_bottom_center 대신 bbox_bottom_center를 쓴다.
    """
    if mode not in MODES:
        raise ValueError(f"알 수 없는 contact mode: {mode}")
    x1, y1, x2, y2 = bbox_img
    if mode == "mask_bottom_center" and mask_polys_img:
        pt = hull_bottom_center(mask_polys_img)
    elif mode == "bbox_center":
        pt = np.array([(x1 + x2) / 2, (y1 + y2) / 2], np.float32)
    else:
        pt = np.array([(x1 + x2) / 2, y2], np.float32)
    return pt, is_truncated(bbox_img, image_size, edge_margin_px)


def contact_map(pt_img: np.ndarray, H: np.ndarray, depth_offset_m: float) -> np.ndarray:
    """접점(영상) -> 맵 좌표. depth_offset_m > 0이면 카메라 반대 방향으로 그만큼 민다.

    마스크 아래 경계는 차 바닥 영역의 "카메라 쪽 끝"이라 칸 앞 경계에 붙는다.
    표준 차 폭의 절반(0.9m)만큼 밀면 차가 어느 방향으로 서 있든 바닥 영역 안쪽에 들어간다.
    영상에서 위쪽 = 바닥에서 카메라 반대 방향이므로, 몇 px 위 점을 함께 투영해 방향을 구한다.
    """
    p = np.asarray(pt_img, np.float32).reshape(2)
    a, b = transform_points(np.stack([p, p - [0, 5]]).astype(np.float32), H)
    if depth_offset_m == 0:
        return a
    v = b - a
    return (a + v / np.linalg.norm(v) * depth_offset_m).astype(np.float32)
