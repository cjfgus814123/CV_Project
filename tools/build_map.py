"""탑뷰 위 칸 생성·클릭·검수 → map.json (CLAUDE.md 5.1-4).

    python tools/build_map.py --lot lotA --cam lotA_cam3

창 2개: "build_map" (탑뷰, 여기서 클릭), "overlay" (맵을 H^-1로 원본 기준 프레임에 되돌려 그린 검수 화면).
조작:
  1 (기본) 칸 줄 모드: 3번 클릭 = 첫 칸 앞 모서리 → 마지막 칸 앞 모서리(줄 끝) → 첫 칸 뒤 모서리
         → 터미널에 "칸수 줄이름 [시작번호]" 입력 (예: "6 A" → A01~A06, "4 B 7" → B07~B10)
  2 다각형 모드: 꼭짓점 클릭 후 Enter → 터미널에 "종류 id" (slot A20 / aisle AISLE1 / no NP1)
  u 되돌리기 (클릭 중이면 마지막 클릭, 아니면 마지막으로 만든 칸·영역)
  s 저장 (map.json, map_check.png, cams/<cam>/map_overlay.png)   q 종료
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from parking.homography import map_polyline_to_img, map_to_bev_matrix, transform_points
from parking.io import load_cam, load_map, save_map
from parking.map_builder import slot_row
from setup_cam import magnifier

KINDS = {"slot": "slots", "aisle": "aisles", "no": "no_parking"}
COL = {"slots": (0, 255, 0), "aisles": (255, 160, 0), "no_parking": (0, 0, 255)}


def draw_items(img, m, to_px, label=True):
    """맵 항목(칸·통로·금지구역)을 그린다. to_px: 맵 좌표 다각형 -> 픽셀 꺾은선 목록."""
    for kind, col in COL.items():
        for it in m[kind]:
            for seg in to_px(it["polygon"]):
                cv2.polylines(img, [np.int32(seg)], False, col, 2, cv2.LINE_AA)
            if label:
                c = to_px([np.mean(it["polygon"], 0)] * 2)
                if c:
                    cv2.putText(img, it["id"], tuple(np.int32(c[0][0]) - [12, -5]), cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, col, 1, cv2.LINE_AA)
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lot", required=True)
    ap.add_argument("--cam", required=True, help="검수 화면에 쓸 cam (그 cam의 bev.png 위에서 클릭)")
    ap.add_argument("--data", default="data")
    a = ap.parse_args()

    lot_dir, cam_dir = Path(a.data) / "lots" / a.lot, Path(a.data) / "cams" / a.cam
    m, cam = load_map(lot_dir), load_cam(cam_dir)
    bev, ref = cv2.imread(str(cam_dir / "bev.png")), cv2.imread(str(cam_dir / "ref_frame.png"))
    M = map_to_bev_matrix(m["bev_px_per_m"], m["bev_origin_m"])
    Minv = np.linalg.inv(M)
    H = cam["H"]
    ref_pt = (ref.shape[1] / 2, ref.shape[0] - 1)

    def bev_px(poly):
        p = transform_points(np.float32(poly), M)
        return [np.vstack([p, p[:1]])] if len(p) > 2 else [p]

    def img_px(poly):
        return map_polyline_to_img(poly, H, ref_pt, closed=len(poly) > 2)

    state = {"click": None, "cursor": None, "enter": False}

    def on_mouse(ev, x, y, *_):
        if 0 <= x < bev.shape[1] and 0 <= y < bev.shape[0]:
            state["cursor"] = (x, y)
            if ev == cv2.EVENT_LBUTTONDOWN:
                state["click"] = (x, y)

    cv2.namedWindow("build_map", cv2.WINDOW_NORMAL)
    cv2.namedWindow("overlay", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("overlay", 1280, 720)
    cv2.setMouseCallback("build_map", on_mouse)
    mode, clicks, history = "row", [], []  # history: (kind, 추가된 개수)

    def overlay():
        return draw_items(ref.copy(), m, img_px)

    cv2.imshow("overlay", overlay())
    while True:
        out = draw_items(bev.copy(), m, bev_px)
        for p in clicks:
            cv2.circle(out, p, 4, (0, 255, 255), -1)
        if len(clicks) >= 2:
            cv2.polylines(out, [np.int32(clicks)], mode == "poly" and len(clicks) > 2, (0, 255, 255), 1)
        magnifier(out, bev, state["cursor"], r=15, z=6)
        n = {k: len(m[k]) for k in COL}
        cur = transform_points(np.float32([state["cursor"]]), Minv)[0] if state["cursor"] else (0, 0)
        lines = [f"mode={mode} [1]row [2]polygon  slots={n['slots']} aisles={n['aisles']} no={n['no_parking']}"
                 f"  cursor=({cur[0]:.2f},{cur[1]:.2f})m",
                 "row: front-first -> front-last -> back-first, then terminal | poly: clicks + Enter",
                 "[u] undo  [s] save  [q] quit"]
        cv2.rectangle(out, (0, 0), (min(out.shape[1], 900), 92), (0, 0, 0), -1)
        for i, t in enumerate(lines):
            cv2.putText(out, t, (10, 26 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("build_map", out)
        key = cv2.waitKey(30) & 0xFF

        if state["click"] is not None:
            clicks.append(state["click"])
            state["click"] = None
        done_poly = mode == "poly" and key in (13, 10) and len(clicks) >= 3
        if (mode == "row" and len(clicks) == 3) or done_poly:
            pm = transform_points(np.float32(clicks), Minv).tolist()
            try:
                if mode == "row":
                    s = input("칸수 줄이름 [시작번호] (예: 6 A), 빈 줄 = 취소: ").split()
                    new = slot_row(*pm, int(s[0]), s[1], int(s[2]) if len(s) > 2 else 1)
                    kind = "slots"
                else:
                    s = input("종류 id (slot A20 / aisle AISLE1 / no NP1), 빈 줄 = 취소: ").split()
                    kind = KINDS[s[0]]
                    new = [{"id": s[1], "polygon": [[round(x, 3), round(y, 3)] for x, y in pm]}]
                    if kind == "slots":
                        new[0]["row"] = s[1].rstrip("0123456789")
                dup = {it["id"] for k in COL for it in m[k]} & {it["id"] for it in new}
                if dup:
                    raise ValueError(f"이미 있는 id: {sorted(dup)}")
                m[kind].extend(new)
                history.append((kind, len(new)))
                w = np.linalg.norm(np.subtract(pm[1], pm[0])) / len(new) if mode == "row" else 0
                print(f"추가: {[it['id'] for it in new]}" + (f" (칸 폭 {w:.2f}m, 길이 "
                      f"{np.linalg.norm(np.subtract(pm[2], pm[0])):.2f}m)" if mode == "row" else ""))
            except (ValueError, IndexError, KeyError) as e:
                print(f"취소 ({e})")
            clicks = []
            cv2.imshow("overlay", overlay())

        if key == ord("q") or cv2.getWindowProperty("build_map", cv2.WND_PROP_VISIBLE) < 1:
            break
        elif key == ord("1"):
            mode, clicks = "row", []
        elif key == ord("2"):
            mode, clicks = "poly", []
        elif key == ord("u"):
            if clicks:
                clicks.pop()
            elif history:
                kind, k = history.pop()
                del m[kind][-k:]
                cv2.imshow("overlay", overlay())
        elif key == ord("s"):
            save_map(lot_dir, m)
            cv2.imwrite(str(lot_dir / "map_check.png"), draw_items(bev.copy(), m, bev_px))
            cv2.imwrite(str(cam_dir / "map_overlay.png"), overlay())
            print(f"저장: {lot_dir}/map.json ({len(m['slots'])}칸), map_check.png, {cam_dir}/map_overlay.png")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
