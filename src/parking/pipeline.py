"""clip 단위 전체 실행 (CLAUDE.md 5.2).

    python -m parking.pipeline --clip data/clips/lotA_cam3_mid
    python -m parking.pipeline --clip data/clips/lotA_cam3_mid --no-video     # CSV만 (빠름)

입력: detections.jsonl (없으면 먼저 검출), data/cams/<cam>/H.json, data/lots/<lot>/map.json
출력: outputs/<clip>/occupancy.csv (frame_idx,time_sec,slot_id,state — 시간 보정 후),
      outputs/<clip>/occupancy_raw.csv (시간 보정 전), outputs/<clip>/split_view.mp4
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon

from parking.contact import away_dir_map, contact_map, contact_point_img
from parking.detect import dedup, load_cache, run_detection
from parking.footprint import footprint_from_det, footprint_states, slot_overlaps
from parking.homography import image_rect, map_polyline_to_img, map_to_bev_matrix, transform_points, visible_region_map
from parking.illegal import StopDetector
from parking.io import load_cam, load_clip_meta, load_config, load_map
from parking.occupancy import (OCCUPIED, UNKNOWN, FREE, apply_visibility, nearest_slot, slot_visibility,
                               slots_from_map, temporal_smooth)
from parking.render import draw_frame_view, draw_map_view, split_view
from parking.track import Tracker


def process_frame(dets: list[dict], H: np.ndarray, slots: dict, cfg: dict,
                  image_size: tuple[int, int]) -> tuple[list[dict], dict[str, int]]:
    """검출 1프레임 -> (접점·배정 칸이 붙은 검출 목록, 칸 상태 {id: 0/1/2}). 가시성·시간 보정 전.

    footprint.enabled면 차 바닥 영역 겹침으로 판정 (2차), 아니면 접점이 든 칸 (1차).
    """
    c, o, fcfg = cfg["contact"], cfg["occupancy"], cfg["footprint"]
    out, states, overlaps = [], dict.fromkeys(slots, FREE), []
    for d in dedup(dets, cfg["model"]["dedup_ios"]):
        pt_img, trunc = contact_point_img(d["mask_polys_img"], d["bbox_img"], c["mode"], image_size, c["edge_margin_px"])
        pt_map = contact_map(pt_img, H, c["depth_offset_m"])
        sid = nearest_slot(pt_map, slots, o["boundary_tol_m"])
        det = {**d, "contact_img": pt_img, "contact_map": pt_map, "slot": sid, "truncated": trunc}
        if fcfg["enabled"] and d["mask_polys_img"]:
            orient = slots[nearest_slot(pt_map, slots, float("inf"))]  # 차 방향 = 가장 가까운 칸 방향
            fp = footprint_from_det(d["mask_polys_img"], H, away_dir_map(pt_img, H), orient, cfg["geometry"]["car_size_m"])
            det["footprint_map"] = np.float32(fp.exterior.coords[:4])
            overlaps.append(slot_overlaps(fp, slots))
        elif sid is not None:
            overlaps.append({sid: 1.0})
        out.append(det)
    if fcfg["enabled"]:
        return out, footprint_states(overlaps, slots, fcfg["tau_occupy"], fcfg["tau_straddle"])
    for ov in overlaps:
        states.update(dict.fromkeys(ov, OCCUPIED))
    return out, states


def run(clip: Path, cfg: dict, out_root: Path, data: Path, video: bool, out_dir: Path | None = None,
        verbose: bool = True) -> Path:
    """out_dir: 결과 위치 (기본 outputs/<clip>). 검출 캐시는 항상 outputs/<clip>/detections.jsonl."""
    meta = load_clip_meta(clip)
    cache = out_root / clip.name / "detections.jsonl"
    out_dir = Path(out_dir) if out_dir else out_root / clip.name
    out_dir.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        run_detection(clip / "video.mp4", cfg, cache, meta.get("valid_sec"))
    recs = load_cache(cache)
    cam, m = load_cam(data / "cams" / meta["cam"]), load_map(data / "lots" / meta["lot"])
    H, image_size = cam["H"], tuple(cam["image_size"])
    slots = slots_from_map(m)
    o = cfg["occupancy"]

    # 가시 영역 (cam마다 고정): ROI가 있으면 ROI, 없으면 영상 네 모서리
    region = np.float64(cam["ground_roi_img"]) if cam.get("ground_roi_img") else image_rect(image_size)
    xs = np.array([p for s in m["slots"] for p in s["polygon"]])
    vis = slot_visibility(slots, visible_region_map(H, region, (*(xs.min(0) - 50), *(xs.max(0) + 50))))
    uncertain = {k for k, v in vis.items() if o["vis_unknown"] <= v < o["vis_judge"]}
    n_unknown = sum(v < o["vis_unknown"] for v in vis.values())
    if verbose:
        print(f"칸 {len(slots)}개: 판정 {len(slots) - n_unknown - len(uncertain)}, 불확실 {len(uncertain)}, "
              f"판단불가 {n_unknown} (커버리지 {1 - n_unknown / len(slots):.0%})")

    zones = {z["id"]: Polygon(z["polygon"]) for z in m.get("aisles", []) + m.get("no_parking", [])}
    ic = cfg["illegal"]
    tracker = Tracker(cfg, image_size) if ic["enabled"] and zones else None
    stops = StopDetector(zones, ic["v_stop_mps"], ic["t_stop_sec"], ic["window_sec"]) if tracker else None

    writer = None
    if video:
        bev = cv2.imread(str(data / "lots" / meta["lot"] / m["bev_image"]))
        M = map_to_bev_matrix(m["bev_px_per_m"], m["bev_origin_m"])
        ref = (image_size[0] / 2, image_size[1] - 1)
        slots_img = {s["id"]: map_polyline_to_img(s["polygon"], H, ref, closed=True) for s in m["slots"]}
        slots_bev = {s["id"]: [transform_points(np.float32(s["polygon"]), M)] for s in m["slots"]}
        zones_bev = [transform_points(np.float32(z.exterior.coords[:-1]), M) for z in zones.values()]
        cap, i_read = cv2.VideoCapture(str(clip / "video.mp4")), -1

    history, t0 = [], time.time()
    with open(out_dir / "occupancy.csv", "w", newline="") as f, open(out_dir / "occupancy_raw.csv", "w", newline="") as fr:
        w, wr = csv.writer(f), csv.writer(fr)
        w.writerow(["frame_idx", "time_sec", "slot_id", "state"])
        wr.writerow(["frame_idx", "time_sec", "slot_id", "state"])
        for n, rec in enumerate(recs):
            dets, raw = process_frame(rec["dets"], H, slots, cfg, image_size)
            raw = apply_visibility(raw, vis, o["vis_unknown"])
            if tracker:  # 3차: 추적 + 구역 안 정지 시간
                for d, tid in zip(dets, tracker.update(dets)):
                    d["track_id"] = tid
                now = stops.update(rec["time_sec"], [(d["track_id"], d["contact_map"]) for d in dets
                                                     if d["track_id"] is not None])
                for d in dets:
                    d["illegal"] = d["track_id"] in now
            history.append(raw)
            states = temporal_smooth(history, o["smooth_n"])
            for sid in slots:
                w.writerow([rec["frame_idx"], rec["time_sec"], sid, states[sid]])
                wr.writerow([rec["frame_idx"], rec["time_sec"], sid, raw[sid]])
            if not video:
                continue
            while i_read < rec["frame_idx"]:  # 순서대로 세며 읽기 (seek 금지)
                cap.grab()
                i_read += 1
            frame = cap.retrieve()[1]
            judged = [s for s in slots if states[s] != UNKNOWN]
            n_free = sum(states[s] == FREE for s in judged)
            header = (f"{clip.name}  t={rec['time_sec']:6.1f}s  free {n_free} / {len(judged)}  "
                      f"(unknown {len(slots) - len(judged)})  cars {len(dets)}"
                      + (f"  illegal {sum(d.get('illegal', False) for d in dets)}" if tracker else ""))
            contacts_bev = transform_points(np.float32([d["contact_map"] for d in dets]).reshape(-1, 2), M) if dets else []
            fps_bev = [transform_points(d["footprint_map"], M) for d in dets if "footprint_map" in d]
            img = split_view(draw_frame_view(frame, dets, slots_img, states, uncertain),
                             draw_map_view(bev, slots_bev, states, contacts_bev, uncertain, fps_bev, zones_bev,
                                           [bool(d.get("illegal")) for d in dets]), header,
                             cfg["render"]["height"])
            if writer is None:
                writer = cv2.VideoWriter(str(out_dir / "split_view.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                         cfg["render"]["video_fps"], (img.shape[1], img.shape[0]))
            writer.write(img)
            if n % 200 == 0:
                print(f"  {n}/{len(recs)} ({time.time() - t0:.0f}s)")
    if writer is not None:
        writer.release()
    if stops:
        events = stops.finish()
        with open(out_dir / "events.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["event_id", "clip_id", "type", "zone_id", "start_sec", "end_sec", "detect_sec", "track_id"])
            for e in events:
                w.writerow([e["event_id"], clip.name, "aisle_parking", e["zone_id"], e["start_sec"], e["end_sec"],
                            e["detect_sec"], e["track_id"]])
        if verbose:
            print(f"불법주차 이벤트 {len(events)}개 -> {out_dir}/events.csv")
    if verbose:
        print(f"완료 {len(recs)}프레임, {time.time() - t0:.0f}s -> {out_dir}/occupancy.csv"
              + (", split_view.mp4" if video else ""))
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="clip 1개 전체 실행 -> 칸 상태 CSV + Split-View 영상")
    ap.add_argument("--clip", required=True, help="data/clips/<clip_id>")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--data", default="data")
    ap.add_argument("--no-video", action="store_true", help="CSV만 만든다")
    a = ap.parse_args()
    run(Path(a.clip), load_config(a.config), Path(a.out), Path(a.data), not a.no_video)


if __name__ == "__main__":
    main()
