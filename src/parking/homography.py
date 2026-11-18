"""H 계산, 재투영 오차, 점·폴리곤 변환, 가시 영역, warp.

점 배열은 np.float32, shape (N,1,2) 규칙 (CLAUDE.md 10장).
"""
from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon


def compute_homography(pts_img: np.ndarray, pts_map: np.ndarray) -> np.ndarray:
    """대응점(6개 이상)으로 영상→맵 H(3x3)를 RANSAC으로 구한다."""
    raise NotImplementedError


def transform_points(pts: np.ndarray, H: np.ndarray) -> np.ndarray:
    """cv2.perspectiveTransform 래퍼. 역방향은 np.linalg.inv(H)를 넘긴다."""
    raise NotImplementedError


def reprojection_error_m(H: np.ndarray, pts_img: np.ndarray, pts_map: np.ndarray) -> dict[str, float]:
    """검증점의 맵 좌표 오차(m). {'mean': ..., 'max': ...}"""
    raise NotImplementedError


def visible_region_map(H: np.ndarray, image_size: tuple[int, int]) -> Polygon:
    """영상 네 모서리를 맵에 투영한 가시 영역 다각형."""
    raise NotImplementedError


def warp_to_bev(frame_img: np.ndarray, H: np.ndarray, px_per_m: float,
                origin_m: tuple[float, float], size_bev: tuple[int, int]) -> np.ndarray:
    """기준 프레임을 탑뷰 이미지(map_bev.png)로 편다."""
    raise NotImplementedError
