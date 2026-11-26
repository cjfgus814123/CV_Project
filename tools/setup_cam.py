"""기준 프레임에서 대응점 클릭 → H.json, ref_frame.png, bev.png (CLAUDE.md 5.1-2,3).

    python tools/setup_cam.py --clip data/clips/lotA_cam3_mid
    python tools/setup_cam.py --clip data/clips/lotA_cam4_mid --auto            # 같은 lot 다른 cam 기준 자동 대응점 -> 화면에서 확인
    python tools/setup_cam.py --clip data/clips/lotA_cam4_mid --auto --no-gui   # 확인 없이 바로 저장 (실시간·일괄 처리용)

--auto: 같은 lot의 다른 cam 기준 프레임과 바닥 특징점(SIFT)을 맞춰 대략 H를 구하고, 그 cam들이 찍은 주차선 모서리를
  새 프레임에서 국소 패치 매칭으로 정밀하게 찾아 대응점으로 넣는다 (parking/autoh.py). 결과는 points.json에 들어가므로
  화면에서 격자·점을 확인하고 틀린 점은 u로 지우거나 직접 추가한 뒤 s로 저장한다. 시점이 많이 다르면 실패 -> 손으로 찍기.

조작 (키 안내는 화면에도 표시):
  클릭      점 추가 → 터미널에 맵 좌표 "x y" (m) 입력 (빈 줄이면 취소)
  v         대응점 / 검증점 모드 전환 (검증점은 H 계산에 안 쓰고 재투영 오차만 잼)
  r         바닥 가시 영역(ROI) 다각형 클릭 모드 전환 (지붕·나무에 가린 곳을 빼는 용도, 선택)
  g         1m 격자 겹쳐 보기 켜기/끄기 (H가 맞으면 격자가 주차선과 나란함)
  m         지도 칸 겹쳐 보기 켜기/끄기 (지금 H로 칸을 그림 -> 주차선과 맞는지 바로 확인)
  점 드래그   이미 찍은 점을 끌어 옮기기 (맵 좌표는 그대로, 위치만 고침)
  오른쪽 클릭 가까운 점 지우기 (틀린 자동 점 골라 지우기)
  u 되돌리기(마지막 점)  s 저장  q 종료
점은 클릭할 때마다 cams/<cam>/points.json에 자동 저장 → 다시 실행하면 이어서 한다.

맵 좌표 정하는 법:
  1) 선이 잘 보이는 한 줄(가까운 줄)에서 줄 방향 = x (선마다 2.5m 간격), 칸 길이 방향 = y (통로 쪽 끝 0, 반대쪽 끝 5).
     선 4~6개의 양 끝 = 8~12점 -> 첫 H. 격자가 그 줄 선과 나란한지 확인.
  2) 다른 줄 점은 커서를 올리면 헤더에 지금 H 기준 맵 좌표가 나온다. 그 값을 표준값으로 반올림해서 입력
     (칸 길이 5.0m, 직각주차 통로 폭 6.0m(주차장법), 칸 폭 2.5m). 화면 양 끝에 넓게 퍼뜨릴수록 좋다.
  3) 넣지 않은 선 끝 2~4개를 v 모드로 검증점으로.
lot의 첫 cam이면 lots/<lot>/map.json 뼈대와 map_bev.png도 만든다 (이후 cam은 같은 맵 좌표계를 쓴다).

같은 lot의 두 번째 cam부터 (팀원용):
  - 첫 4점은 다른 cam의 cams/<cam>/points.png (찍은 점과 좌표가 적힌 그림)에서 같은 모서리를 찾아 입력한다.
  - 4점 이상이면 다른 cam들이 쓴 좌표가 화면에 하늘색 +로 나온다 (추천점). 그 근처를 클릭하면
    터미널에 추천 좌표가 나오고, Enter만 누르면 그 좌표로 들어간다 (x = 취소, 직접 입력도 가능).
저장하면 points.png (찍은 점 확인)와, 맵에 칸이 있으면 map_overlay.png (맵 칸을 원본에 되돌려 그림)도 만든다.
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
from parking.autoh import auto_correspondences, load_refs, lot_polygon
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


def reference_points(data: Path, lot: str, cam: str) -> np.ndarray:
    """같은 lot의 다른 cam들이 H.json에 쓴 맵 좌표 (중복 제거), (K,2). 팀원이 같은 좌표계로 점을 찍게 돕는다."""
    out = []
    for f in sorted((data / "cams").glob("*/H.json")):
        h = json.loads(f.read_text(encoding="utf-8"))
        if h.get("lot_id") == lot and h.get("cam_id") != cam:
            out += [tuple(round(v, 3) for v in p["map"]) for p in h["correspondences"] + h["validation"]]
    map_path = data / "lots" / lot / "map.json"
    if map_path.exists():  # 지도의 칸 모서리 (선 중심선 교차점)도 추천
        out += [tuple(round(v, 3) for v in q) for s in json.loads(map_path.read_text(encoding="utf-8"))["slots"]
                for q in s["polygon"]]
    return np.array(sorted(set(out)), np.float64).reshape(-1, 2)


def nearest_point(pts: dict, xy, r: float = 12.0):
    """xy(영상 px)에서 r 안의 가장 가까운 대응점·검증점 -> (종류, 번호) 또는 None."""
    best = None
    for k in ("corr", "val"):
        for i, p in enumerate(pts[k]):
            d = float(np.hypot(p["img"][0] - xy[0], p["img"][1] - xy[1]))
            if d <= r and (best is None or d < best[0]):
                best = (d, k, i)
    return best[1:] if best else None


def suggest(xy_img, H, refs: np.ndarray, max_m: float = 1.0):
    """클릭 위치를 지금 H로 맵에 옮겨 가장 가까운 추천 좌표 (max_m 안에 없으면 None)."""
    if H is None or not len(refs):
        return None
    pm = transform_points(np.float32([xy_img]), H)[0]
    d = np.linalg.norm(refs - pm, axis=1)
    return refs[d.argmin()] if d.min() <= max_m else None


def draw(frame, pts, H, mode, show_grid, cursor, res, refs=None, slots_map=None):
    out = frame.copy()
    h, w = frame.shape[:2]
    ref = (w / 2, h - 1)
    if H is not None and slots_map:  # 지도 칸을 지금 H로 영상에 그림
        for sid, poly in slots_map:
            pieces = map_polyline_to_img(poly, H, ref, closed=True)
            for p in pieces:
                cv2.polylines(out, [np.int32(p)], True, (0, 255, 0), 1, cv2.LINE_AA)
            if pieces:
                c = np.concatenate(pieces).mean(0)
                cv2.putText(out, sid, (int(c[0]) - 12, int(c[1]) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (0, 255, 255), 1, cv2.LINE_AA)
    if H is not None and refs is not None:  # 추천점: 다른 cam이 쓴 좌표가 이 화면에서 있어야 할 자리
        used = {tuple(p["map"]) for p in pts["corr"] + pts["val"]}
        for X, Y in refs:
            if (X, Y) in used:
                continue
            q = map_polyline_to_img([[X, Y], [X, Y]], H, ref)
            if q and 0 <= q[0][0][0] < w and 0 <= q[0][0][1] < h:
                u = tuple(np.int32(q[0][0]))
                cv2.drawMarker(out, u, (255, 255, 0), cv2.MARKER_CROSS, 14, 2)
                cv2.putText(out, f"{X:g},{Y:g}", (u[0] + 6, u[1] + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (255, 255, 0), 1, cv2.LINE_AA)
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
             "click: add point (Enter = suggested +)  drag: move point  right-click: delete  [m] slots [g] grid [u] undo [s] save [q] quit",
             "red line = click -> where H puts its map coord (short = good)"]
    cv2.rectangle(out, (0, 0), (1400, 92), (0, 0, 0), -1)
    for i, t in enumerate(lines):
        cv2.putText(out, t, (10, 26 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def points_sheet(frame: np.ndarray, pts: dict, res) -> np.ndarray:
    """팀원용 안내 그림: 찍은 점에 번호, 오른쪽에 번호별 맵 좌표 목록 (초록 = 대응점, 주황 = 검증점)."""
    out = frame.copy()
    items = [("corr", p) for p in pts["corr"]] + [("val", p) for p in pts["val"]]
    for i, (k, p) in enumerate(items, 1):
        u = tuple(np.int32(p["img"]))
        cv2.circle(out, u, 7, COL[k], -1)
        cv2.circle(out, u, 7, (0, 0, 0), 1)
        for col, th in (((0, 0, 0), 5), (COL[k], 2)):
            cv2.putText(out, str(i), (u[0] + 9, u[1] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, th, cv2.LINE_AA)
    panel = np.zeros((out.shape[0], 330, 3), np.uint8)
    cv2.putText(panel, "no.  map x, y (m)", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    for i, (k, p) in enumerate(items, 1):
        err = f"  e={res[i - 1]:.2f}" if k == "corr" and res is not None else "  (val)"
        cv2.putText(panel, f"{i:2d}   {p['map'][0]:g}, {p['map'][1]:g}{err}", (12, 34 + 30 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, COL[k], 1, cv2.LINE_AA)
    return np.hstack([out, panel])


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
    cv2.imwrite(str(cam_dir / "points.png"), points_sheet(frame, pts, res))
    if m["slots"]:  # 맵 칸을 이 cam의 원본에 되돌려 그림 -> 실제 주차선과 맞는지 확인
        ov, ref = frame.copy(), (w / 2, h - 1)
        for s in m["slots"]:
            pieces = map_polyline_to_img(s["polygon"], H, ref, closed=True)
            for p in pieces:
                cv2.polylines(ov, [np.int32(p)], True, (0, 255, 0), 2, cv2.LINE_AA)
            if pieces:
                c = np.concatenate(pieces).mean(0)
                cv2.putText(ov, s["id"], (int(c[0]) - 15, int(c[1]) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(cam_dir / "map_overlay.png"), ov)
    ok = val is not None and val["max"] <= 0.15
    print(f"저장: {cam_dir}/H.json, ref_frame.png, bev.png, points.png"
          + (", map_overlay.png" if m["slots"] else "")
          + f" | 대응점 오차 평균 {res.mean():.3f}m 최대 {res.max():.3f}m"
          + (f" | 검증 평균 {val['mean']:.3f}m 최대 {val['max']:.3f}m" if val else "")
          + (" -> 좋음" if ok else " -> 검증 최대 오차가 0.15m를 넘거나 검증점이 없음: 점을 더 넓게 찍거나 틀린 점을 u로 지우기"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clip", required=True, help="data/clips/<clip_id> (가능하면 빈 상태)")
    ap.add_argument("--sec", type=float, default=None, help="기준 프레임 시각 (기본: valid_sec 시작 + 1초)")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data", default="data")
    ap.add_argument("--auto", action="store_true", help="같은 lot 다른 cam 기준으로 대응점 자동 추정 (기존 points.json 덮어씀)")
    ap.add_argument("--no-gui", action="store_true", help="--auto 결과를 화면 확인 없이 바로 저장")
    ap.add_argument("--frame", default=None, help="기준 프레임 이미지 파일 (영상 대신, 평가용)")
    a = ap.parse_args()

    cfg, meta = load_config(a.config), load_clip_meta(a.clip)
    cam_dir, lot_dir = Path(a.data) / "cams" / meta["cam"], Path(a.data) / "lots" / meta["lot"]
    cam_dir.mkdir(parents=True, exist_ok=True)
    video = Path(a.clip) / "video.mp4"
    fps = cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FPS)
    sec = a.sec if a.sec is not None else meta.get("valid_sec", [0])[0] + 1.0
    frame_idx = round(sec * fps)
    frame = cv2.imread(a.frame) if a.frame else read_frame(video, frame_idx)
    h, w = frame.shape[:2]

    pts_path = cam_dir / "points.json"
    pts = json.loads(pts_path.read_text()) if pts_path.exists() else {"corr": [], "val": [], "roi": []}
    if pts.get("frame_idx", frame_idx) != frame_idx:
        print(f"주의: points.json은 프레임 {pts['frame_idx']} 기준 (지금 {frame_idx}). 카메라가 고정이면 괜찮음")
    pts["frame_idx"] = frame_idx
    if a.auto:
        ac = cfg["auto_h"]
        m = json.loads((lot_dir / "map.json").read_text(encoding="utf-8"))
        corr, info = auto_correspondences(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                          load_refs(Path(a.data), meta["lot"], meta["cam"]),
                                          lot_polygon(m, ac["lot_buffer_m"]), ac)
        print(f"자동 대응점: 기준 cam 일치점 {info['inliers']}, 기준 {info.get('ref')}, "
              f"찾은 점 {info.get('found', 0)} -> 사용 {len(corr)}" + (f" | 실패: {info['reason']}" if "reason" in info else ""))
        if corr:
            print("주의: 검증점도 자동으로 찾은 점이라 재투영 오차가 실제보다 작게 나온다 (lotA 시험: 표시 최대 0.07m, "
                  "손 점 기준 최대 0.29~0.89m). map_overlay.png에서 칸이 주차선과 맞는지 눈으로 확인할 것")
            pts["corr"] = [p for i, p in enumerate(corr) if i % ac["val_every"] != ac["val_every"] - 1]
            pts["val"] = [p for i, p in enumerate(corr) if i % ac["val_every"] == ac["val_every"] - 1]
            pts_path.write_text(json.dumps(pts, indent=1))
        elif a.no_gui:
            raise SystemExit("자동 추정 실패: --auto 없이 손으로 찍기")
        if a.no_gui:
            fitted = fit(pts, cfg)
            if not fitted:
                raise SystemExit("대응점이 4개 미만")
            save_all(a, cfg, meta, frame, frame_idx, pts, fitted, cam_dir, lot_dir)
            return
    print(f"{meta['cam']}: 프레임 {frame_idx} ({sec:.1f}s), 이어서 시작: {len(pts['corr'])}+{len(pts['val'])}점")
    refs = reference_points(Path(a.data), meta["lot"], meta["cam"])
    if len(refs):
        print(f"추천점 {len(refs)}개 (같은 lot의 다른 cam 좌표). 첫 4점은 그 cam의 points.png를 보고 입력")

    state = {"click": None, "cursor": None, "drag": None, "rclick": None, "moved": False}
    map_path = lot_dir / "map.json"
    slots_map = [(s["id"], s["polygon"]) for s in json.loads(map_path.read_text(encoding="utf-8"))["slots"]] \
        if map_path.exists() else []

    def on_mouse(ev, x, y, *_):
        if not (0 <= x < w and 0 <= y < h):
            return
        state["cursor"] = (x, y)
        if ev == cv2.EVENT_LBUTTONDOWN:
            hit = nearest_point(pts, (x, y)) if mode != "roi" else None
            if hit:  # 기존 점 위에서 누르면 드래그
                state["drag"] = hit
            else:
                state["click"] = (x, y)
        elif ev == cv2.EVENT_MOUSEMOVE and state["drag"]:
            k, i = state["drag"]
            pts[k][i]["img"] = [float(x), float(y)]
            state["moved"] = True
        elif ev == cv2.EVENT_LBUTTONUP and state["drag"]:
            state["drag"] = None
            pts_path.write_text(json.dumps(pts, indent=1))
        elif ev == cv2.EVENT_RBUTTONDOWN:
            state["rclick"] = (x, y)

    win = "setup_cam"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1600, 900)
    cv2.setMouseCallback(win, on_mouse)
    mode, show_grid, show_slots = "corr", True, bool(slots_map)
    fitted = fit(pts, cfg)
    while True:
        if state["moved"]:  # 드래그 중: H를 바로 다시 계산 (칸·격자가 따라 움직임)
            state["moved"] = False
            fitted = fit(pts, cfg)
        if state["rclick"] is not None:
            hit, state["rclick"] = nearest_point(pts, state["rclick"], 20.0), None
            if hit:
                p = pts[hit[0]].pop(hit[1])
                print(f"지움: {hit[0]} ({p['map'][0]:g}, {p['map'][1]:g})")
                pts_path.write_text(json.dumps(pts, indent=1))
                fitted = fit(pts, cfg)
        H, res = (fitted[0], fitted[1]) if fitted else (None, None)
        cv2.imshow(win, draw(frame, pts, H, mode, show_grid, state["cursor"], res, refs,
                             slots_map if show_slots else None))
        key = cv2.waitKey(30) & 0xFF
        if state["click"] is not None:
            xy, state["click"] = [float(v) for v in state["click"]], None
            if mode == "roi":
                pts["roi"].append(xy)
            else:
                sg = suggest(xy, H, refs)
                hint = f"Enter = 추천 ({sg[0]:g} {sg[1]:g}), x = 취소" if sg is not None else "빈 줄 = 취소"
                s = input(f"[{mode}] 클릭 ({xy[0]:.0f},{xy[1]:.0f})의 맵 좌표 'x y' ({hint}): ").strip()
                if not s and sg is not None:
                    s = f"{sg[0]} {sg[1]}"
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
        elif key == ord("m"):
            show_slots = not show_slots
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
