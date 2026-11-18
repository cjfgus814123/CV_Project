"""체커보드 캘리브레이션, undistort (CLAUDE.md 5.1-1). 광각(0.5배) 렌즈일 때 필수.

캘리브레이션은 "폰 + 렌즈 + 해상도" 단위다 (장소와 무관). 결과는 data/calib/<이름>/calib.json.

    python -m parking.calibrate --video data/calib/phone_0.5x/checkerboard.mp4 \\
        --pattern 8 6 --square-mm 20 --preview data/clips/lotA_cam1_full/video.mp4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def board_points(pattern: tuple[int, int], square_m: float) -> np.ndarray:
    """체커보드 내부 코너의 3D 좌표 (z=0), (cols*rows, 3) float32."""
    cols, rows = pattern
    g = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_m
    return np.hstack([g, np.zeros((len(g), 1))]).astype(np.float32)


def find_corners(video_path: str | Path, pattern: tuple[int, int], every: int) -> tuple[list[np.ndarray], list[int], tuple[int, int]]:
    """영상을 every 프레임마다 읽어 코너를 찾는다. 반환: (코너 목록, 프레임 번호, (w, h))."""
    cap = cv2.VideoCapture(str(video_path))
    corners, idxs, i, size = [], [], 0, None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % every == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            size = gray.shape[::-1]
            found, c = cv2.findChessboardCornersSB(gray, pattern, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
            if found:
                corners.append(c.reshape(-1, 1, 2).astype(np.float32))
                idxs.append(i)
        i += 1
    cap.release()
    if size is None:
        raise ValueError(f"영상을 읽을 수 없음: {video_path}")
    return corners, idxs, size


def coverage_grid(corners: list[np.ndarray], image_size: tuple[int, int], n: int = 6) -> np.ndarray:
    """영상을 n×n 칸으로 나눠 코너가 한 번이라도 찍힌 칸 = True. 가장자리가 비면 왜곡 추정이 부정확."""
    w, h = image_size
    grid = np.zeros((n, n), bool)
    for c in corners:
        p = c.reshape(-1, 2)
        gx = np.clip((p[:, 0] / w * n).astype(int), 0, n - 1)
        gy = np.clip((p[:, 1] / h * n).astype(int), 0, n - 1)
        grid[gy, gx] = True
    return grid


def calibrate(corners: list[np.ndarray], pattern: tuple[int, int], square_m: float,
              image_size: tuple[int, int], fix_k3: bool = True) -> dict:
    """내부 파라미터 K, 왜곡 계수 dist, RMS 재투영 오차(px).

    fix_k3: 고차 항 k3를 0으로 고정. 체커보드가 가장자리를 덜 덮으면 k3가 가장자리에서 발산하므로 기본 True.
    """
    if len(corners) < 5:
        raise ValueError(f"체커보드가 검출된 프레임이 너무 적음 ({len(corners)}장, 최소 5장)")
    obj = [board_points(pattern, square_m)] * len(corners)
    flags = cv2.CALIB_FIX_K3 if fix_k3 else 0
    rms, K, dist, _, _ = cv2.calibrateCamera(obj, corners, image_size, None, None, flags=flags)
    return {"image_size": list(image_size), "K": K.tolist(), "dist": dist.ravel().tolist(), "rms_px": float(rms)}


def undistort_maps(calib: dict, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """remap용 맵과 새 K. alpha=1: 원본 픽셀을 모두 남김(가장자리 검은 영역), 0: 검은 영역 없게 자름."""
    K, dist = np.array(calib["K"]), np.array(calib["dist"])
    size = tuple(calib["image_size"])
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, size, alpha, size)
    m1, m2 = cv2.initUndistortRectifyMap(K, dist, None, new_K, size, cv2.CV_16SC2)
    return m1, m2, new_K


def undistort(frame: np.ndarray, maps: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    return cv2.remap(frame, maps[0], maps[1], cv2.INTER_LINEAR)


def _first_frame(video: str | Path, frame_idx: int = 0) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    for _ in range(frame_idx + 1):
        ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"프레임을 읽을 수 없음: {video}")
    return frame


def main() -> None:
    from parking.io import load_config

    ap = argparse.ArgumentParser(description="체커보드 영상 -> calib.json")
    ap.add_argument("--video", required=True)
    ap.add_argument("--pattern", type=int, nargs=2, required=True, metavar=("COLS", "ROWS"), help="내부 코너 수")
    ap.add_argument("--square-mm", type=float, required=True)
    ap.add_argument("--every", type=int, default=10, help="N프레임마다 검사")
    ap.add_argument("--max-views", type=int, default=40)
    ap.add_argument("--out", default=None, help="기본: 영상과 같은 폴더/calib.json")
    ap.add_argument("--preview", default=None, help="이 영상의 첫 프레임을 원본|보정 나란히 저장")
    ap.add_argument("--config", default="configs/default.yaml")
    a = ap.parse_args()

    pattern = tuple(a.pattern)
    corners, idxs, size = find_corners(a.video, pattern, a.every)
    print(f"체커보드 검출: {len(corners)}장 (검사 프레임 {a.every}개마다)")
    if len(corners) > a.max_views:  # 비슷한 연속 프레임이 많으면 느리고 치우침 -> 고르게 솎음
        keep = np.linspace(0, len(corners) - 1, a.max_views).round().astype(int)
        corners, idxs = [corners[k] for k in keep], [idxs[k] for k in keep]
    grid = coverage_grid(corners, size)
    print("화면 커버리지 (6x6, #=코너 있음):")
    for row in grid:
        print("  " + "".join("#" if v else "." for v in row))
    edge = np.concatenate([grid[0], grid[-1], grid[1:-1, 0], grid[1:-1, -1]]).mean()
    print(f"커버리지 {grid.mean():.0%}, 가장자리 칸 {edge:.0%}")
    if edge < 0.5:
        print("경고: 가장자리 커버리지 < 50%. 영상 가장자리의 보정값을 믿을 수 없음 -> 체커보드를 네 모서리까지 움직여 재촬영 권장")

    cfg = load_config(a.config)["calibration"]
    calib = calibrate(corners, pattern, a.square_mm / 1000, size, cfg["fix_k3"])
    calib.update({"source_video": str(a.video), "pattern": list(pattern), "square_mm": a.square_mm,
                  "frames_used": idxs, "coverage": float(grid.mean()), "edge_coverage": float(edge),
                  "fix_k3": cfg["fix_k3"]})
    out = Path(a.out) if a.out else Path(a.video).with_name("calib.json")
    out.write_text(json.dumps(calib, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"RMS 재투영 오차 {calib['rms_px']:.3f}px, dist {np.round(calib['dist'], 4).tolist()} -> {out}")

    if a.preview:
        alpha = cfg["alpha"]
        frame = _first_frame(a.preview)
        both = np.hstack([frame, undistort(frame, undistort_maps(calib, alpha))])
        p = out.with_name("preview_undistort.jpg")
        cv2.imwrite(str(p), cv2.resize(both, None, fx=0.5, fy=0.5))
        print(f"미리보기 (왼쪽 원본 | 오른쪽 보정): {p}")


if __name__ == "__main__":
    main()
