"""합성 데이터: 바닥을 비스듬히 내려다보는 가상 핀홀 카메라 (H를 정확히 안다)."""
import numpy as np
import pytest

IMG_W, IMG_H = 1920, 1080


def make_camera(height=8.0, pitch_deg=35.0, f=1000.0, cam_xy=(0.0, 0.0)):
    """맵 좌표: x 오른쪽, y 카메라 앞쪽, 바닥 z=0. 카메라는 +y를 보며 pitch만큼 숙임.

    반환: dict(K, R, t, H) — H는 영상->맵.
    """
    p = np.deg2rad(pitch_deg)
    K = np.array([[f, 0, IMG_W / 2], [0, f, IMG_H / 2], [0, 0, 1]])
    x_c = np.array([1.0, 0, 0])
    z_c = np.array([0, np.cos(p), -np.sin(p)])
    y_c = np.cross(z_c, x_c)
    R = np.stack([x_c, y_c, z_c])
    C = np.array([cam_xy[0], cam_xy[1], height])
    t = -R @ C
    H_map2img = K @ np.column_stack([R[:, 0], R[:, 1], t])
    return {"K": K, "R": R, "t": t, "H": np.linalg.inv(H_map2img)}


def project(cam, pts3d):
    """3D 점 (N,3) -> 영상 픽셀 (N,2)."""
    pc = (cam["R"] @ np.asarray(pts3d, float).T).T + cam["t"]
    uv = (cam["K"] @ pc.T).T
    return uv[:, :2] / uv[:, 2:3]


@pytest.fixture
def cam():
    return make_camera()


def slot_rows():
    """가짜 map: 카메라 앞 A줄(y 10~15), 뒤 B줄(y 15~20), 칸 2.5 x 5.0m, 줄마다 4칸 (x -5~5)."""
    from shapely.geometry import box
    slots = {}
    for row, y0 in (("A", 10.0), ("B", 15.0)):
        for i in range(4):
            x0 = -5.0 + 2.5 * i
            slots[f"{row}{i + 1:02d}"] = box(x0, y0, x0 + 2.5, y0 + 5.0)
    return slots


def car_mask_poly(cam, center_xy, size=(1.8, 4.7), height=1.5):
    """바닥 중심 center_xy, y축 방향으로 주차한 직육면체 차 -> 영상 마스크 다각형(볼록 껍질), 박스."""
    import cv2
    cx, cy = center_xy
    w, l = size
    corners = [(cx + sx * w / 2, cy + sy * l / 2, z) for sx in (-1, 1) for sy in (-1, 1) for z in (0, height)]
    uv = project(cam, corners).astype(np.float32)
    hull = cv2.convexHull(uv).reshape(-1, 2)
    x1, y1 = uv.min(0)
    x2, y2 = uv.max(0)
    return [hull.tolist()], [float(x1), float(y1), float(x2), float(y2)]
