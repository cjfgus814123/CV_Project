"""기준 프레임에서 대응점 클릭 → H.json, ref_frame.png, bev.png (CLAUDE.md 5.1-2,3).

    python tools/setup_cam.py --clip data/clips/lotA_cam3_mid

조작 (키 안내는 화면에도 표시):
  클릭      점 추가 → 터미널에 맵 좌표 "x y" (m) 입력 (빈 줄이면 취소)
  v         대응점 / 검증점 모드 전환 (검증점은 H 계산에 안 쓰고 재투영 오차만 잼)
  r         바닥 가시 영역(ROI) 다각형 클릭 모드 전환 (지붕·나무에 가린 곳을 빼는 용도, 선택)
  g         1m 격자 겹쳐 보기 켜기/끄기 (H가 맞으면 격자가 주차선과 나란함)
  u 되돌리기  s 저장  q 종료
점은 클릭할 때마다 cams/<cam>/points.json에 자동 저장 → 다시 실행하면 이어서 한다.

맵 좌표 정하는 법:
  1) 선이 잘 보이는 한 줄(가까운 줄)에서 줄 방향 = x (선마다 2.5m 간격), 칸 길이 방향 = y (통로 쪽 끝 0, 반대쪽 끝 5).
     선 4~6개의 양 끝 = 8~12점 -> 첫 H. 격자가 그 줄 선과 나란한지 확인.
  2) 다른 줄 점은 커서를 올리면 헤더에 지금 H 기준 맵 좌표가 나온다. 그 값을 표준값으로 반올림해서 입력
     (칸 길이 5.0m, 직각주차 통로 폭 6.0m(주차장법), 칸 폭 2.5m). 화면 양 끝에 넓게 퍼뜨릴수록 좋다.
  3) 넣지 않은 선 끝 2~4개를 v 모드로 검증점으로.
lot의 첫 cam이면 lots/<lot>/map.json 뼈대와 map_bev.png도 만든다 (이후 cam은 같은 맵 좌표계를 쓴다).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import box

from parking.homography import (compute_homography, image_rect, map_polyline_to_img, reprojection_error_m,
                                transform_points, visible_region_map, warp_to_bev)
from parking.io import load_clip_meta, load_config, read_frame, save_cam, save_map

MODES = ("corr", "val", "roi")
COL = {"corr": (0, 255, 0), "val": (0, 165, 255), "roi": (255, 0, 255)}


def magnifier(out: np.ndarray, src: np.ndarray, cursor, r: int = 20, z: int = 6) -> None:
    """커서 주변 (2r)^2 영역을 z배로 키워 out 오른쪽 위에 그린다 (모서리를 정확히 찍기 위함)."""
    if cursor is None:
        return
    cx, cy = cursor
    patch = cv2.copyMakeBorder(src, r, r, r, r, cv2.BORDER_CONSTANT)[cy:cy + 2 * r, cx:cx + 2 * r]
    patch = cv2.resize(patch, (2 * r * z, 2 * r * z), interpolation=cv2.INTER_NEAREST)
    cv2.line(patch, (r * z, 0), (r * z, 2 * r * z), (0, 0, 255), 1)
    cv2.line(patch, (0, r * z), (2 * r * z, r * z), (0, 0, 255), 1)
    w = out.shape[1]
    out[100:100 + 2 * r * z, w - 2 * r * z - 10:w - 10] = patch


def fit(pts: dict, cfg: dict):
    """대응점 4개 이상이면 (H, 대응점 오차 목록, 검증 오차 dict|None), 아니면 None."""
    c = pts["corr"]
    if len(c) < 4:
        return None
    img, mp = np.float32([p["img"] for p in c]), np.float32([p["map"] for p in c])
    H, _ = compute_homography(img, mp, cfg["homography"]["ransac_thresh_m"])
    res = np.linalg.norm(transform_points(img, H) - mp, axis=1)
    v = pts["val"]
    val = reprojection_error_m(H, np.float32([p["img"] for p in v]), np.float32([p["map"] for p in v])) if v else None
    return H, res, val


def draw(frame, pts, H, mode, show_grid, cursor, res):
    out = frame.copy()
    h, w = frame.shape[:2]
    ref = (w / 2, h - 1)
    if H is not None and show_grid:
        allm = np.array([p["map"] for p in pts["corr"] + pts["val"]])
        (x0, y0), (x1, y1) = np.floor(allm.min(0)) - 10, np.ceil(allm.max(0)) + 10
        for x in np.arange(x0, x1 + 1):
            for seg in map_polyline_to_img([[x, y0], [x, y1]], H, ref):
                cv2.polylines(out, [np.int32(seg)], False, (255, 255, 0) if x % 5 == 0 else (160, 160, 0), 1, cv2.LINE_AA)
        for y in np.arange(y0, y1 + 1):
            for seg in map_polyline_to_img([[x0, y], [x1, y]], H, ref):
                cv2.polylines(out, [np.int32(seg)], False, (255, 255, 0) if y % 5 == 0 else (160, 160, 0), 1, cv2.LINE_AA)
    if len(pts["roi"]) >= 2:
        cv2.polylines(out, [np.int32(pts["roi"])], len(pts["roi"]) > 2, COL["roi"], 2)
    for k in ("corr", "val"):
        for i, p in enumerate(pts[k]):
            u = tuple(np.int32(p["img"]))
            if H is not None:  # 맵 좌표를 H^-1로 되돌린 위치 = 이 점이 "있어야 할" 자리
                q = map_polyline_to_img([p["map"], p["map"]], H, ref)
                if q:
                    cv2.line(out, u, tuple(np.int32(q[0][0])), (0, 0, 255), 2)
            cv2.circle(out, u, 5, COL[k], -1)
            err = f" e={res[i]:.2f}" if k == "corr" and res is not None else ""
            cv2.putText(out, f"({p['map'][0]:g},{p['map'][1]:g}){err}", (u[0] + 7, u[1] - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL[k], 1, cv2.LINE_AA)
    magnifier(out, frame, cursor)
    cur = ""
    if H is not None and cursor is not None:  # 지금 H로 본 커서의 맵 좌표 -> 다른 줄의 점 좌표를 읽을 때 사용
        cm = transform_points(np.float32([cursor]), H)[0]
        cur = f"  cursor=({cm[0]:.2f},{cm[1]:.2f})m"
    lines = [f"mode={mode} (v: corr/val, r: roi)  corr={len(pts['corr'])} val={len(pts['val'])} roi={len(pts['roi'])}{cur}",
             "click: add point -> type 'x y' in terminal   [g] grid  [u] undo  [s] save  [q] quit",
             "red line = click -> where H puts its map coord (short = good)"]
    cv2.rectangle(out, (0, 0), (1250, 92), (0, 0, 0), -1)
    for i, t in enumerate(lines):
        cv2.putText(out, t, (10, 26 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def save_all(a, cfg, meta, frame, frame_idx, pts, fitted, cam_dir: Path, lot_dir: Path):
    H, res, val = fitted
    h, w = frame.shape[:2]
    if val is None:
        print("주의: 검증점이 없어 재투영 오차를 알 수 없음 (v 모드로 2~4개 추가 권장)")
    cv2.imwrite(str(cam_dir / "ref_frame.png"), frame)
    save_cam(cam_dir, {
        "cam_id": meta["cam"], "lot_id": meta["lot"], "ref_clip": Path(a.clip).name, "ref_frame_idx": frame_idx,
        "undistort": False, "image_size": [w, h],
        "correspondences": pts["corr"], "validation": pts["val"], "ground_roi_img": pts["roi"] or None,
        "H": H, "fit_error_m": {"mean": float(res.mean()), "max": float(res.max())}, "reproj_error_m": val,
    })
    map_path = lot_dir / "map.json"
    if map_path.exists():  # 맵 좌표계는 lot 첫 cam이 정함 -> 같은 탑뷰 범위로 펴서 비교
        m = json.loads(map_path.read_text(encoding="utf-8"))
        px, origin, size = m["bev_px_per_m"], m["bev_origin_m"], m["bev_size_px"]
    else:
        px = cfg["geometry"]["bev_px_per_m"]
        allm = np.array([p["map"] for p in pts["corr"] + pts["val"]])
        mg = cfg["homography"]["bev_margin_m"]
        region = np.float64(pts["roi"]) if len(pts["roi"]) >= 3 else image_rect((w, h))
        vis = visible_region_map(H, region, (*(allm.min(0) - mg), *(allm.max(0) + mg)))
        x0, y0, x1, y1 = vis.bounds
        origin = [float(np.floor(x0)), float(np.floor(y0))]
        size = [int(np.ceil((x1 - origin[0]) * px)), int(np.ceil((y1 - origin[1]) * px))]
        m = {"lot_id": meta["lot"], "units": "m", "slot_size": cfg["geometry"]["slot_size_m"],
             "bev_image": "map_bev.png", "bev_px_per_m": px, "bev_origin_m": origin, "bev_size_px": size,
             "bev_cam": meta["cam"], "landmarks": [], "slots": [], "aisles": [], "no_parking": []}
    bev = warp_to_bev(frame, H, px, origin, size)
    cv2.imwrite(str(cam_dir / "bev.png"), bev)
    if not map_path.exists():
        save_map(lot_dir, m)
        shutil.copy(cam_dir / "bev.png", lot_dir / "map_bev.png")
        print(f"lot 맵 뼈대 생성: {map_path}, {lot_dir / 'map_bev.png'} ({size[0]}x{size[1]}px)")
    print(f"저장: {cam_dir}/H.json, ref_frame.png, bev.png | 대응점 오차 평균 {res.mean():.3f}m 최대 {res.max():.3f}m"
          + (f" | 검증 평균 {val['mean']:.3f}m 최대 {val['max']:.3f}m" if val else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clip", required=True, help="data/clips/<clip_id> (가능하면 빈 상태)")
    ap.add_argument("--sec", type=float, default=None, help="기준 프레임 시각 (기본: valid_sec 시작 + 1초)")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data", default="data")
    a = ap.parse_args()

    cfg, meta = load_config(a.config), load_clip_meta(a.clip)
    cam_dir, lot_dir = Path(a.data) / "cams" / meta["cam"], Path(a.data) / "lots" / meta["lot"]
    cam_dir.mkdir(parents=True, exist_ok=True)
    video = Path(a.clip) / "video.mp4"
    fps = cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FPS)
    sec = a.sec if a.sec is not None else meta.get("valid_sec", [0])[0] + 1.0
    frame_idx = round(sec * fps)
    frame = read_frame(video, frame_idx)
    h, w = frame.shape[:2]

    pts_path = cam_dir / "points.json"
    pts = json.loads(pts_path.read_text()) if pts_path.exists() else {"corr": [], "val": [], "roi": []}
    if pts.get("frame_idx", frame_idx) != frame_idx:
        print(f"주의: points.json은 프레임 {pts['frame_idx']} 기준 (지금 {frame_idx}). 카메라가 고정이면 괜찮음")
    pts["frame_idx"] = frame_idx
    print(f"{meta['cam']}: 프레임 {frame_idx} ({sec:.1f}s), 이어서 시작: {len(pts['corr'])}+{len(pts['val'])}점")

    state = {"click": None, "cursor": None}

    def on_mouse(ev, x, y, *_):
        if 0 <= x < w and 0 <= y < h:
            state["cursor"] = (x, y)
            if ev == cv2.EVENT_LBUTTONDOWN:
                state["click"] = (x, y)

    win = "setup_cam"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1600, 900)
    cv2.setMouseCallback(win, on_mouse)
    mode, show_grid = "corr", True
    fitted = fit(pts, cfg)
    while True:
        H, res = (fitted[0], fitted[1]) if fitted else (None, None)
        cv2.imshow(win, draw(frame, pts, H, mode, show_grid, state["cursor"], res))
        key = cv2.waitKey(30) & 0xFF
        if state["click"] is not None:
            xy, state["click"] = [float(v) for v in state["click"]], None
            if mode == "roi":
                pts["roi"].append(xy)
            else:
                s = input(f"[{mode}] 클릭 ({xy[0]:.0f},{xy[1]:.0f})의 맵 좌표 'x y' (빈 줄 = 취소): ").strip()
                try:
                    X, Y = (float(v) for v in s.replace(",", " ").split())
                except ValueError:
                    print("취소")
                    continue
                pts[mode].append({"img": xy, "map": [X, Y]})
            pts_path.write_text(json.dumps(pts, indent=1))
            fitted = fit(pts, cfg)
            if fitted:
                print(f"  대응점 오차 평균 {fitted[1].mean():.3f}m 최대 {fitted[1].max():.3f}m"
                      + (f", 검증 평균 {fitted[2]['mean']:.3f}m" if fitted[2] else ""))
        if key == ord("q") or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        elif key == ord("v"):
            mode = "val" if mode == "corr" else "corr"
        elif key == ord("r"):
            mode = "corr" if mode == "roi" else "roi"
        elif key == ord("g"):
            show_grid = not show_grid
        elif key == ord("u") and pts[mode]:
            pts[mode].pop()
            pts_path.write_text(json.dumps(pts, indent=1))
            fitted = fit(pts, cfg)
        elif key == ord("s"):
            if fitted:
                save_all(a, cfg, meta, frame, frame_idx, pts, fitted, cam_dir, lot_dir)
            else:
                print("대응점이 4개 이상이어야 저장 가능")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
