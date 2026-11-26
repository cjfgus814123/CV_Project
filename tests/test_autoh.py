"""autoh: 무늬 있는 바닥을 두 시점으로 찍은 합성 영상에서, 기준 cam의 대응점으로 새 cam의 H를 되찾는다."""
import cv2
import numpy as np

from parking.autoh import auto_correspondences, lot_polygon
from parking.homography import transform_points

PX = 20  # 바닥 무늬 이미지: 1m = 20px, 30m x 20m
S = np.array([[PX, 0, 0], [0, PX, 0], [0, 0, 1]], np.float64)  # 맵(m) -> 무늬 픽셀
CFG = {"sift_features": 4000, "ratio": 0.75, "ransac_px": 3.0, "min_inliers": 30, "patch_px": 20,
       "search_px": 16, "min_texture": 5, "min_score": 0.6, "ransac_m": 0.3}


def ground_texture() -> np.ndarray:
    rng = np.random.default_rng(0)
    tex = cv2.GaussianBlur((rng.random((400, 600)) * 255).astype(np.uint8), (0, 0), 2)
    for x in np.arange(0, 30.1, 2.5):  # 주차선
        cv2.line(tex, (int(x * PX), 0), (int(x * PX), 400), 255, 3)
    cv2.line(tex, (0, 100), (600, 100), 255, 3)
    return tex


def view(tex, quad_img):
    """무늬의 네 모서리가 quad_img로 가는 카메라 -> (영상, H 영상->맵)."""
    A = cv2.getPerspectiveTransform(np.float32([[0, 0], [600, 0], [600, 400], [0, 400]]), np.float32(quad_img))
    return cv2.warpPerspective(tex, A, (960, 540)), np.linalg.inv(A @ S)


def test_auto_correspondences_recover_new_view():
    tex = ground_texture()
    img_ref, H_ref = view(tex, [[200, 120], [760, 120], [940, 520], [20, 520]])
    img_new, H_new = view(tex, [[150, 100], [700, 140], [900, 530], [60, 500]])
    grid = np.array([[x, y] for x in np.arange(2.5, 28, 2.5) for y in (2.5, 5.0, 10.0, 15.0)], np.float32)
    ref_img_pts = transform_points(grid, np.linalg.inv(H_ref))
    refs = [{"cam": "ref", "img": img_ref, "H": H_ref,
             "points": [{"img": p.tolist(), "map": m.tolist()} for p, m in zip(ref_img_pts, grid)]}]
    lot = lot_polygon({"slots": [{"polygon": [[0, 0], [30, 0], [30, 20], [0, 20]]}]}, 0.0)
    corr, info = auto_correspondences(img_new, refs, lot, CFG)
    assert info["ref"] == "ref" and len(corr) >= 20
    H, _ = cv2.findHomography(np.float32([p["img"] for p in corr]), np.float32([p["map"] for p in corr]))
    new_img_pts = transform_points(grid, np.linalg.inv(H_new))  # 정답 위치
    err = np.linalg.norm(transform_points(new_img_pts, H) - grid, axis=1)
    assert err.max() < 0.1


def test_auto_correspondences_fail_on_unrelated_image():
    tex = ground_texture()
    img_ref, H_ref = view(tex, [[200, 120], [760, 120], [940, 520], [20, 520]])
    noise = (np.random.default_rng(1).random((540, 960)) * 255).astype(np.uint8)
    refs = [{"cam": "ref", "img": img_ref, "H": H_ref, "points": []}]
    lot = lot_polygon({"slots": [{"polygon": [[0, 0], [30, 0], [30, 20], [0, 20]]}]}, 0.0)
    corr, info = auto_correspondences(noise, refs, lot, CFG)
    assert corr == [] and "reason" in info
