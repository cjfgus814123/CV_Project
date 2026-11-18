"""H 계산, 재투영 오차, 점·폴리곤 변환, 가시 영역, warp.

점 배열은 np.float32. 입력은 (N,2) 또는 (N,1,2) 모두 받는다 (CLAUDE.md 10장).
H는 항상 영상(_img) -> 맵(_map, m) 방향. 반대 방향은 np.linalg.inv(H).
"""
from __future__ import annotations

import cv2
import numpy as np
from shapely.geometry import Polygon, box


def _as_cv(pts: np.ndarray) -> np.ndarray:
    return np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)


def compute_homography(pts_img: np.ndarray, pts_map: np.ndarray,
                       ransac_thresh_m: float) -> tuple[np.ndarray, np.ndarray]:
    """대응점으로 영상->맵 H(3x3)를 RANSAC으로 구한다.

    ransac_thresh_m은 맵 좌표 기준(m). 반환: (H, inlier 여부 bool 배열 (N,)).
    """
    src, dst = _as_cv(pts_img), _as_cv(pts_map)
    if len(src) != len(dst):
        raise ValueError(f"대응점 개수 불일치: img {len(src)}, map {len(dst)}")
    if len(src) < 4:
        raise ValueError(f"대응점이 4개 이상 필요 (현재 {len(src)}개)")
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh_m)
    if H is None:
        raise ValueError("H 계산 실패: 점들이 한 직선 위에 몰려 있지 않은지 확인")
    return H, mask.ravel().astype(bool)


def transform_points(pts: np.ndarray, H: np.ndarray) -> np.ndarray:
    """점을 H로 변환한다. 출력 shape = 입력 shape (float32)."""
    a = np.asarray(pts, dtype=np.float32)
    out = cv2.perspectiveTransform(a.reshape(-1, 1, 2), np.asarray(H, dtype=np.float64))
    return out.reshape(a.shape)


def reprojection_error_m(H: np.ndarray, pts_img: np.ndarray, pts_map: np.ndarray) -> dict[str, float]:
    """검증점을 H로 옮긴 위치와 실제 맵 좌표의 거리(m). {'mean', 'max'}"""
    pred = transform_points(pts_img, H).reshape(-1, 2)
    d = np.linalg.norm(pred - np.asarray(pts_map, dtype=np.float32).reshape(-1, 2), axis=1)
    return {"mean": float(d.mean()), "max": float(d.max())}


def map_to_bev_matrix(px_per_m: float, origin_m: tuple[float, float]) -> np.ndarray:
    """맵 좌표(m) -> 탑뷰 이미지 픽셀. origin_m이 탑뷰 이미지의 (0,0) 픽셀."""
    s = float(px_per_m)
    ox, oy = origin_m
    return np.array([[s, 0, -ox * s], [0, s, -oy * s], [0, 0, 1]], dtype=np.float64)


def warp_to_bev(frame_img: np.ndarray, H: np.ndarray, px_per_m: float,
                origin_m: tuple[float, float], size_bev: tuple[int, int]) -> np.ndarray:
    """프레임을 탑뷰 이미지로 편다. size_bev = (width, height) 픽셀."""
    M = map_to_bev_matrix(px_per_m, origin_m) @ np.asarray(H, dtype=np.float64)
    return cv2.warpPerspective(frame_img, M, tuple(size_bev))


def _clip_below_horizon(poly_img: np.ndarray, H: np.ndarray) -> np.ndarray:
    """영상 다각형에서 지평선 너머(H로 보내면 무한대·뒤집힘) 부분을 잘라낸다.

    w(x,y) = H[2]·(x,y,1)이 0이 되는 직선이 지평선. 다각형의 가장 아래 점(바닥이라고 가정)과
    같은 부호인 쪽만 남긴다 (Sutherland-Hodgman, 직선 1개).
    """
    w = poly_img @ H[2, :2] + H[2, 2]
    sign = np.sign(w[np.argmax(poly_img[:, 1])]) or 1.0
    w = w * sign
    eps = 1e-3 * w.max()  # 지평선 바로 아래는 맵에서 너무 멀어 수치가 불안정 -> 살짝 띄움
    out = []
    n = len(poly_img)
    for i in range(n):
        p, q, wp, wq = poly_img[i], poly_img[(i + 1) % n], w[i], w[(i + 1) % n]
        if wp >= eps:
            out.append(p)
        if (wp >= eps) != (wq >= eps):
            out.append(p + (q - p) * (eps - wp) / (wq - wp))
    return np.array(out, dtype=np.float64).reshape(-1, 2)


def image_rect(image_size: tuple[int, int]) -> np.ndarray:
    """영상 네 모서리 (w, h) -> (4,2)."""
    w, h = image_size
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float64)


def visible_region_map(H: np.ndarray, region_img: np.ndarray,
                       bounds_map: tuple[float, float, float, float]) -> Polygon:
    """영상 위 영역(기본: image_rect)을 맵에 투영한 가시 영역.

    지평선 너머는 잘라내고, 맵 범위 bounds_map=(xmin, ymin, xmax, ymax)와 교집합을 취한다.
    """
    H = np.asarray(H, dtype=np.float64)
    clipped = _clip_below_horizon(np.asarray(region_img, dtype=np.float64).reshape(-1, 2), H)
    if len(clipped) < 3:
        return Polygon()
    poly = Polygon(transform_points(clipped, H).astype(np.float64))
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly.intersection(box(*bounds_map))
