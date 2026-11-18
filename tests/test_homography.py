import cv2
import numpy as np
import pytest
from shapely.geometry import Point

from conftest import IMG_H, IMG_W, make_camera
from parking.homography import (compute_homography, image_rect, map_to_bev_matrix,
                                reprojection_error_m, transform_points, visible_region_map,
                                warp_to_bev)

GRID_MAP = np.array([[x, y] for x in (-5, 0, 5) for y in (8, 12, 16)], np.float32)  # 9점


def to_img(cam, pts_map):
    return transform_points(pts_map, np.linalg.inv(cam["H"]))


def test_round_trip_and_shape(cam):
    pts = np.random.default_rng(0).uniform([-5, 8], [5, 20], (20, 2)).astype(np.float32)
    back = transform_points(to_img(cam, pts), cam["H"])
    assert np.allclose(back, pts, atol=1e-3)
    assert transform_points(pts.reshape(-1, 1, 2), cam["H"]).shape == (20, 1, 2)


def test_compute_recovers_H(cam):
    H, inl = compute_homography(to_img(cam, GRID_MAP), GRID_MAP, 0.15)
    assert inl.all()
    test = np.array([[2.5, 10.0], [-3.0, 18.0]], np.float32)
    assert np.allclose(transform_points(to_img(cam, test), H), test, atol=1e-2)


def test_ransac_rejects_outlier(cam):
    img = to_img(cam, GRID_MAP).copy()
    img[4] += 60  # 클릭 실수 1개
    H, inl = compute_homography(img, GRID_MAP, 0.15)
    assert not inl[4] and inl.sum() == 8
    assert reprojection_error_m(H, np.delete(img, 4, 0), np.delete(GRID_MAP, 4, 0))["max"] < 0.02


def test_too_few_points(cam):
    with pytest.raises(ValueError):
        compute_homography(to_img(cam, GRID_MAP[:3]), GRID_MAP[:3], 0.15)


def test_reprojection_error(cam):
    shifted = GRID_MAP.copy()
    shifted[0, 0] += 0.3
    err = reprojection_error_m(cam["H"], to_img(cam, GRID_MAP), shifted)
    assert err["max"] == pytest.approx(0.3, abs=1e-3)


def test_visible_region_clips_horizon():
    cam = make_camera(pitch_deg=20)  # 영상 윗부분이 지평선 너머(하늘)
    vis = visible_region_map(cam["H"], image_rect((IMG_W, IMG_H)), (-30, -10, 30, 60))
    assert vis.is_valid and not vis.is_empty
    near = transform_points(np.float32([[IMG_W / 2, IMG_H - 20]]), cam["H"])[0]
    assert vis.contains(Point(near))
    assert not vis.contains(Point(0, -5))  # 카메라 뒤
    assert vis.bounds[3] == pytest.approx(60)  # 먼 쪽은 맵 범위에서 잘림


def test_map_to_bev_matrix():
    M = map_to_bev_matrix(20, (-10, 0))
    assert np.allclose(transform_points(np.float32([[-10, 0], [-9, 1]]), M), [[0, 0], [20, 20]])


def test_warp_to_bev(cam):
    px, origin, size = 20, (-10.0, 0.0), (400, 600)
    bev = np.zeros((size[1], size[0]), np.uint8)
    M = map_to_bev_matrix(px, origin)
    cv2.rectangle(bev, (180, 200), (220, 260), 255, -1)  # 맵 x -1..1, y 10..13
    frame = cv2.warpPerspective(bev, np.linalg.inv(M @ cam["H"]), (IMG_W, IMG_H))
    back = warp_to_bev(frame, cam["H"], px, origin, size)
    ys, xs = np.nonzero(back > 127)
    assert abs(xs.mean() - 200) < 2 and abs(ys.mean() - 230) < 2
