"""3D 박스 추정: 합성 카메라 (높이 11m, 비스듬히 내려다봄)."""
import numpy as np

from parking.box3d import bottom_polygon, box_corners, camera_matrix, fit_box, project

K = np.array([[936.0, 0, 960], [0, 936, 540], [0, 0, 1]])
SIZE = (1.8, 4.7, 1.5)


def _true_P():
    # 카메라: 맵 (0, -8, 11)에서 +y 방향을 35도 내려다봄. 맵 z = 위
    a = np.radians(35)
    R = np.array([[1, 0, 0], [0, -np.sin(a), -np.cos(a)], [0, np.cos(a), -np.sin(a)]])  # 행 = 카메라 x, y, z 축
    C = np.array([0, -8, 11.0])
    return K @ np.column_stack([R, -R @ C])


def test_camera_from_homography():
    P = _true_P()
    H = np.linalg.inv(P[:, [0, 1, 3]])  # 영상 -> 맵 바닥
    P2 = camera_matrix(H, K)
    pts = np.array([[3, 10, 0], [5, 12, 1.5], [-2, 15, 0.8]], float)
    assert np.allclose(project(P2, pts), project(P, pts), atol=0.5)


def test_fit_recovers_angled_car():
    P = _true_P()
    xy, th = np.array([4.0, 13.0]), np.radians(90 + 15)  # 칸(길이 = +y, 90도)에서 15도 돌아간 차
    hull = project(P, box_corners(xy[None], np.array([th]), SIZE))[0]
    fxy, fth, iou = fit_box([hull.tolist()], P, xy + [0.6, -0.5], np.radians(90), SIZE, [-30, -15, 0, 15, 30, 90])
    assert iou > 0.95
    assert np.linalg.norm(fxy - xy) < 0.15 and abs(np.degrees(fth - th)) < 3
    assert abs(bottom_polygon(fxy, fth, SIZE).area - 1.8 * 4.7) < 1e-6


def test_slot_volume_front_slot_hides_back_slot():
    # 카메라(y=-8)에 가까운 칸 A의 상자가 뒤 칸 B 상자 아래쪽을 가린다 -> B의 안 가려진 영역은 원래보다 작다
    from shapely.geometry import box
    from parking.slot_volume import visible_regions, volume_states
    P = _true_P()
    slots = {"A": box(-1.25, 5, 1.25, 10), "B": box(-1.25, 10, 1.25, 15)}
    reg = visible_regions(slots, P, 1.5)
    full_b = visible_regions({"B": slots["B"]}, P, 1.5)["B"]
    assert reg["A"].area > 0 and reg["B"].area < full_b.area * 0.9
    car_in_b = project(P, box_corners(np.array([[0.0, 12.5]]), np.array([np.pi / 2]), SIZE))[0]
    from shapely.geometry import MultiPoint
    st, cov = volume_states([[list(MultiPoint(car_in_b).convex_hull.exterior.coords)]], reg, 0.4)
    assert st == {"A": 0, "B": 1}
