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
