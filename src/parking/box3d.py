"""(2차 대안) 단일 시점 3D 박스 추정으로 차 바닥 영역 구하기.

BoxCars(Sochor 외)·Kocur 외처럼, 보정된 고정 카메라에서 차를 바닥 위에 놓인 직육면체로 보고
투영 모양이 마스크와 가장 잘 맞는 위치·방향을 찾는다. 소실점 대신 바닥 H와 내부 파라미터 K로 카메라를 복원한다.

- 카메라: H^-1 (맵 -> 영상) = λ K [r1 r2 t]. r3 = r1 x r2 (맵 z축 = 위쪽).
- 박스: 크기 (폭, 길이, 높이) 고정, 중심 (x, y)와 방향 θ를 찾는다.
- 점수: 박스 8꼭짓점 투영의 볼록 껍질과 마스크 볼록 껍질의 IoU (cv2.intersectConvexConvex, 빠름).
- 결과: 박스 밑면 = 차 바닥 영역 (맵 좌표 사각형). 칸 겹침 판정은 footprint.py 규칙을 그대로 쓴다.
"""
from __future__ import annotations

import cv2
import numpy as np
from shapely.geometry import Polygon


def camera_matrix(H: np.ndarray, K: np.ndarray) -> np.ndarray:
    """영상->맵 H와 내부 파라미터 K -> 투영 행렬 P (3x4, 맵 좌표 (x, y, z) -> 영상). z는 바닥에서 위로 (m)."""
    B = np.linalg.inv(K) @ np.linalg.inv(H)
    lam = 2 / (np.linalg.norm(B[:, 0]) + np.linalg.norm(B[:, 1]))
    if B[2, 2] * lam < 0:  # 카메라 앞쪽 (t_z > 0)
        lam = -lam
    r1, r2, t = lam * B[:, 0], lam * B[:, 1], lam * B[:, 2]
    U, _, Vt = np.linalg.svd(np.stack([r1, r2, np.cross(r1, r2)], 1))
    R = U @ Vt
    P = K @ np.column_stack([R, t])
    # z+가 영상에서 위쪽(v 감소)이 되도록 (맵이 왼손 좌표계면 r1 x r2가 아래를 가리킴)
    a, b = P @ [0, 0, 0, 1], P @ [0, 0, 1, 1]
    if b[1] / b[2] > a[1] / a[2]:
        P = K @ np.column_stack([R[:, 0], R[:, 1], -R[:, 2], t])
    return P


def box_corners(xy: np.ndarray, theta: np.ndarray, size: tuple[float, float, float]) -> np.ndarray:
    """후보 N개의 박스 꼭짓점 (N, 8, 3). xy (N,2), theta (N,) = 길이 방향 각도 (rad), size = (폭, 길이, 높이)."""
    w, l, h = size
    base = np.array([[-w, -l], [w, -l], [w, l], [-w, l]]) / 2  # (4,2) 폭 x, 길이 y
    c, s = np.cos(theta), np.sin(theta)
    # 길이 방향 = (c, s), 폭 방향 = (s, -c)
    pts = (base[None, :, 0:1] * np.stack([s, -c], -1)[:, None, :]
           + base[None, :, 1:2] * np.stack([c, s], -1)[:, None, :] + xy[:, None, :])  # (N,4,2)
    z0, z1 = np.zeros(pts.shape[:2] + (1,)), np.full(pts.shape[:2] + (1,), h)
    return np.concatenate([np.concatenate([pts, z0], -1), np.concatenate([pts, z1], -1)], 1)


def project(P: np.ndarray, pts3: np.ndarray) -> np.ndarray:
    """(..., 3) 맵 3D 점 -> (..., 2) 영상 픽셀."""
    q = pts3 @ P[:, :3].T + P[:, 3]
    return q[..., :2] / q[..., 2:3]


def _iou(hull_a: np.ndarray, area_a: float, pts_b: np.ndarray) -> float:
    hb = cv2.convexHull(pts_b.astype(np.float32))
    inter, _ = cv2.intersectConvexConvex(hull_a, hb)
    ab = cv2.contourArea(hb)
    return inter / (area_a + ab - inter) if inter > 0 else 0.0


def _search(P, hull, area, xy, thetas, offsets, size):
    cand_xy = (xy[None, :] + offsets).repeat(len(thetas), 0)
    cand_th = np.tile(thetas, len(offsets))
    img = project(P, box_corners(cand_xy, cand_th, size))
    scores = np.array([_iou(hull, area, p) for p in img])
    i = int(scores.argmax())
    return cand_xy[i], cand_th[i], float(scores[i])


def fit_box(mask_polys_img: list, P: np.ndarray, init_xy: np.ndarray, slot_theta: float,
            size: tuple[float, float, float], angles_deg: list[float],
            rotate_gain: float = 0.0) -> tuple[np.ndarray, float, float]:
    """마스크에 가장 잘 맞는 박스 (중심 xy, 방향 θ, IoU). init_xy 근처, 칸 방향 + angles_deg 후보에서 찾는다.

    거친 탐색 (±1m, 0.5m 간격) -> 고운 탐색 (±0.2m, 0.1m 간격).
    rotate_gain: 칸 방향 박스보다 IoU가 이만큼 이상 높아야 돌린 박스를 쓴다 (주차된 차는 대부분 칸 방향.
    차 모양이 직육면체가 아니라 조금 돌린 박스가 IoU가 약간 높게 나오는 경우가 많음). 칸 방향이면 각도는 고정.
    """
    pts = np.concatenate([np.asarray(p, np.float32).reshape(-1, 2) for p in mask_polys_img])
    hull = cv2.convexHull(pts)
    area = cv2.contourArea(hull)
    g = np.arange(-1.0, 1.01, 0.5)
    coarse = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
    f = np.arange(-0.2, 0.21, 0.1)
    fine = np.stack(np.meshgrid(f, f), -1).reshape(-1, 2)
    init = np.asarray(init_xy, float)
    xy, th, _ = _search(P, hull, area, init, np.array([slot_theta]), coarse, size)
    aligned = _search(P, hull, area, xy, np.array([slot_theta]), fine, size)
    rot = [a for a in angles_deg if a != 0]
    if not rot:
        return aligned
    xy, th, _ = _search(P, hull, area, init, slot_theta + np.radians(rot), coarse, size)
    rotated = _search(P, hull, area, xy, th + np.radians([-4, -2, 0, 2, 4]), fine, size)
    return rotated if rotated[2] > aligned[2] + rotate_gain else aligned


def bottom_polygon(xy: np.ndarray, theta: float, size: tuple[float, float, float]) -> Polygon:
    """박스 밑면 (맵 사각형)."""
    return Polygon(box_corners(np.asarray(xy, float)[None], np.array([theta]), size)[0, :4, :2])
