"""GT 라벨링: 칸 클릭으로 0→1→2→-1 순환, data/clips/<clip>/gt_occupancy.csv 저장 (CLAUDE.md 7).

    python tools/sample_frames.py --clip data/clips/lotA_cam3_mid --every-sec 10   # 먼저 프레임 뽑기
    python tools/label_gt.py --clip data/clips/lotA_cam3_mid

- 시스템 예측(검출·판정)은 보여주지 않는다 (공정한 평가).
- 새 프레임은 직전 프레임의 라벨을 복사해서 시작 → 바뀐 칸만 클릭하면 된다.
- 클릭한 변경은 뒤 프레임에도 이어서 적용된다 (그 칸이 같은 상태였던 프레임까지). 차가 나간 프레임에서 다시 클릭하면 된다.
- 칸 안쪽 바닥(칸 이름 근처)을 클릭한다. 차 지붕을 클릭하면 뒤 칸이 눌릴 수 있다.
상태: 0 빈칸(초록) / 1 점유(빨강) / 2 침범=선을 걸친 차가 있는 칸(노랑) / -1 판단불가=가려져 안 보임(회색)
키: 클릭 다음 상태, d/→ 다음 프레임, a/← 이전, u 되돌리기, m 칸 색 켜기/끄기, s 저장, q 저장 후 종료
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

from parking.homography import image_rect, map_polyline_to_img, transform_points, visible_region_map
from parking.io import load_cam, load_clip_meta, load_config, load_map
from parking.occupancy import UNKNOWN, nearest_slot, slot_visibility, slots_from_map
from parking.render import STATE_COLOR

sys.path.insert(0, str(Path(__file__).parent))
from setup_cam import magnifier  # noqa: E402

CYCLE = {0: 1, 1: 2, 2: -1, -1: 0}


def load_gt(path: Path) -> dict[int, dict[str, int]]:
    gt: dict[int, dict[str, int]] = {}
    if path.exists():
        for r in csv.DictReader(path.open()):
            gt.setdefault(int(r["frame_idx"]), {})[r["slot_id"]] = int(r["state"])
    return gt


def save_gt(path: Path, clip_id: str, gt: dict[int, dict[str, int]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "frame_idx", "slot_id", "state"])
        for fi in sorted(gt):
            for sid in sorted(gt[fi]):
                w.writerow([clip_id, fi, sid, gt[fi][sid]])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clip", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--config", default="configs/default.yaml")
    a = ap.parse_args()

    clip = Path(a.clip)
    meta = load_clip_meta(clip)
    cam, m = load_cam(Path(a.data) / "cams" / meta["cam"]), load_map(Path(a.data) / "lots" / meta["lot"])
    H, slots = cam["H"], slots_from_map(m)
    ref = (cam["image_size"][0] / 2, cam["image_size"][1] - 1)
    slots_img = {s["id"]: map_polyline_to_img(s["polygon"], H, ref, closed=True) for s in m["slots"]}
    # 첫 프레임 기본값: 화면에 거의 안 보이는 칸(가시 비율 < vis_unknown)은 -1, 나머지는 0
    region = np.float64(cam["ground_roi_img"]) if cam.get("ground_roi_img") else image_rect(cam["image_size"])
    xs = np.array([p for s in m["slots"] for p in s["polygon"]])
    vis = slot_visibility(slots, visible_region_map(H, region, (*(xs.min(0) - 50), *(xs.max(0) + 50))))
    init = {k: (UNKNOWN if v < load_config(a.config)["occupancy"]["vis_unknown"] else 0) for k, v in vis.items()}
    frames = sorted(clip.glob("gt_frames/f*.jpg"))
    if not frames:
        sys.exit("gt_frames가 없음: 먼저 tools/sample_frames.py 실행")
    fidx = [int(p.stem[1:]) for p in frames]
    gt_path = clip / "gt_occupancy.csv"
    gt = load_gt(gt_path)
    print(f"{len(frames)}프레임, 이미 라벨 {sum(f in gt for f in fidx)}개")

    state = {"click": None, "cursor": None}

    def on_mouse(ev, x, y, *_):
        state["cursor"] = (x, y)
        if ev == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (x, y)

    win = "label_gt"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1600, 900)
    cv2.setMouseCallback(win, on_mouse)
    k = next((i for i, f in enumerate(fidx) if f not in gt), 0)  # 첫 미라벨 프레임부터
    undo: list[tuple[int, str, list[tuple[int, int]]]] = []  # (현재 k, 칸, [(프레임, 이전 상태)])
    show, img_k, img = True, None, None
    while True:
        f = fidx[k]
        if f not in gt:  # 직전 라벨 프레임 복사, 없으면 기본값
            prev = [g for g in fidx[:k] if g in gt]
            gt[f] = dict(gt[prev[-1]]) if prev else dict(init)
        if img_k != k:
            img, img_k = cv2.imread(str(frames[k])), k
        labels = gt[f]
        out = img.copy()
        if show:
            over = out.copy()
            for sid, pieces in slots_img.items():
                for p in pieces:
                    cv2.fillPoly(over, [np.int32(p)], STATE_COLOR[labels.get(sid, UNKNOWN)])
            out = cv2.addWeighted(over, 0.25, out, 0.75, 0)
            for sid, pieces in slots_img.items():
                col = STATE_COLOR[labels.get(sid, UNKNOWN)]
                for p in pieces:
                    cv2.polylines(out, [np.int32(p)], True, col, 2, cv2.LINE_AA)
                c = np.concatenate(pieces).mean(0)
                cv2.putText(out, sid, (int(c[0]) - 14, int(c[1]) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)
        magnifier(out, img, state["cursor"], r=40, z=3)
        cnt = {s: sum(v == s for v in labels.values()) for s in (0, 1, 2, -1)}
        lines = [f"frame {k + 1}/{len(frames)} (idx {f})  free {cnt[0]}  occ {cnt[1]}  straddle {cnt[2]}  unknown {cnt[-1]}",
                 "click: 0 free -> 1 occupied -> 2 straddle -> -1 unknown   (applies to this and following frames)",
                 "[d/->] next  [a/<-] prev  [u] undo  [m] colors  [s] save  [q] save+quit"]
        cv2.rectangle(out, (0, 0), (1120, 92), (0, 0, 0), -1)
        for i, t in enumerate(lines):
            cv2.putText(out, t, (10, 26 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(win, out)
        key = cv2.waitKeyEx(30)

        if state["click"] is not None:
            pm = transform_points(np.float32([state["click"]]), H)[0]
            state["click"] = None
            sid = nearest_slot(pm, slots, 0.0)
            if sid is not None:  # 이 프레임부터 같은 상태가 이어지는 뒤 프레임까지 함께 바꾼다
                old, changed = labels[sid], []
                for g in fidx[k:]:
                    if g not in gt or gt[g].get(sid) != old:
                        break
                    changed.append((g, old))
                    gt[g][sid] = CYCLE[old]
                undo.append((k, sid, changed))
        if key in (ord("q"), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        elif key in (ord("d"), 65363):
            k = min(k + 1, len(frames) - 1)
        elif key in (ord("a"), 65361):
            k = max(k - 1, 0)
        elif key == ord("u") and undo:
            k, sid, changed = undo.pop()
            for g, old in changed:
                gt[g][sid] = old
        elif key == ord("m"):
            show = not show
        elif key == ord("s"):
            save_gt(gt_path, clip.name, gt)
            print(f"저장: {gt_path} ({len(gt)}프레임)")
    save_gt(gt_path, clip.name, gt)
    print(f"저장: {gt_path} ({len(gt)}프레임)")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
