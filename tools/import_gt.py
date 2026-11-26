"""Drive에서 받은 GT를 제자리에 넣고 검사 (team_guide.md 부록).

    python tools/import_gt.py --from ~/Downloads/2_완료 --event-clips data/clips/lotA_cam7_t*

- 칸 GT: 파일 안의 clip_id 열로 clip을 찾아 data/clips/<clip>/gt_occupancy.csv 로 복사.
  같은 clip을 여러 사람이 했으면 (`<clip>__<이름>.csv`) 첫 사람 것을 gt_occupancy.csv, 나머지는
  gt_occupancy__<이름>.csv 로 두고 사람 간 일치도를 출력.
- 불법주차 GT: 머리글에 event_id가 있는 CSV를 clip_id별로 나눠 data/clips/<clip>/gt_events.csv.
  --event-clips에 준 clip 중 사건이 없는 clip은 머리글만 있는 gt_events.csv (= 사건 없음, 오경보 평가용).
- 검사: gt_frames의 사진이 모두 라벨됐는지, 칸 이름, 상태 값(-1/0/1/2), 사건 종류·구역·시각.
이미 있는 GT는 --force 없이는 덮어쓰지 않는다 (덮어쓸 때는 .bak로 보관).
"""
from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path

from parking.io import load_clip_meta, load_map

EVENT_HEADER = ["event_id", "clip_id", "type", "zone_id", "start_sec", "end_sec"]


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as f:  # Google 시트 CSV의 BOM 대비
        return list(csv.DictReader(f))


def _put(src_rows: list[dict], header: list[str], dst: Path, force: bool) -> bool:
    if dst.exists() and not force:
        print(f"  건너뜀: {dst} 이미 있음 (--force로 덮어쓰기)")
        return False
    if dst.exists():
        shutil.copy(dst, dst.with_suffix(dst.suffix + ".bak"))
    with dst.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        w.writerows(src_rows)
    return True


def check_occupancy(clip: Path, rows: list[dict], slot_ids: set[str]) -> list[str]:
    """문제 목록 (빈 목록 = 통과)."""
    probs = []
    frames = {int(p.stem[1:]) for p in (clip / "gt_frames").glob("f*.jpg")}
    by_frame = defaultdict(dict)
    for r in rows:
        by_frame[int(r["frame_idx"])][r["slot_id"]] = r["state"]
    if missing := sorted(frames - set(by_frame)):
        probs.append(f"라벨 안 된 사진 {len(missing)}장 (마지막 사진까지 d로 넘겼는지): {missing[:5]}...")
    if extra := sorted(set(by_frame) - frames):
        probs.append(f"gt_frames에 없는 프레임 {len(extra)}개: {extra[:5]}")
    for f, s in by_frame.items():
        if set(s) != slot_ids:
            probs.append(f"프레임 {f}: 칸 이름이 지도와 다름 (빠짐 {sorted(slot_ids - set(s))[:3]}, 모름 {sorted(set(s) - slot_ids)[:3]})")
            break
    if bad := {r["state"] for r in rows} - {"-1", "0", "1", "2"}:
        probs.append(f"잘못된 상태 값: {bad}")
    return probs


def agreement(a: list[dict], b: list[dict]) -> str:
    ka = {(r["frame_idx"], r["slot_id"]): r["state"] for r in a}
    kb = {(r["frame_idx"], r["slot_id"]): r["state"] for r in b}
    common = [k for k in ka if k in kb and ka[k] != "-1" and kb[k] != "-1"]
    if not common:
        return "공통 판정 없음"
    same = sum(ka[k] == kb[k] for k in common)
    diff = defaultdict(int)
    for k in common:
        if ka[k] != kb[k]:
            diff[f"{ka[k]}->{kb[k]}"] += 1
    return f"일치 {same}/{len(common)} = {same / len(common):.3f}, 불일치 {dict(diff)}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from", dest="src", required=True, help="Drive에서 받은 CSV 폴더")
    ap.add_argument("--event-clips", nargs="*", default=[], help="불법주차 GT를 확인한 clip (사건 없으면 빈 gt_events.csv)")
    ap.add_argument("--data", default="data")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    data, clips_dir = Path(a.data), Path(a.data) / "clips"
    occ, events = defaultdict(list), []
    for p in sorted(Path(a.src).glob("*.csv")):
        rows = _read(p)
        if not rows:
            continue
        if "event_id" in rows[0]:
            events += [r for r in rows if r.get("event_id")]
            continue
        name = p.stem.split("__", 1)[1] if "__" in p.stem else "?"
        for cid in sorted({r["clip_id"] for r in rows}):
            occ[cid].append((name, [r for r in rows if r["clip_id"] == cid]))

    n_ok = 0
    for cid, versions in sorted(occ.items()):
        clip = clips_dir / cid
        if not clip.exists():
            print(f"[{cid}] clip 폴더 없음 -> 건너뜀")
            continue
        meta = load_clip_meta(clip)
        slot_ids = {s["id"] for s in load_map(data / "lots" / meta["lot"])["slots"]}
        for k, (name, rows) in enumerate(versions):
            probs = check_occupancy(clip, rows, slot_ids)
            dst = clip / ("gt_occupancy.csv" if k == 0 else f"gt_occupancy__{name}.csv")
            put = _put(rows, ["clip_id", "frame_idx", "slot_id", "state"], dst, a.force)
            n_frames = len({r["frame_idx"] for r in rows})
            print(f"[{cid}] {name}: {n_frames}프레임 -> {dst.name if put else '(안 넣음)'} | "
                  + ("검사 통과" if not probs else "문제: " + " / ".join(probs)))
            n_ok += put and not probs
        if len(versions) > 1:
            print(f"[{cid}] 사람 간 일치도 ({versions[0][0]} vs {versions[1][0]}): {agreement(versions[0][1], versions[1][1])}")

    by_clip = defaultdict(list)
    for r in events:
        by_clip[r["clip_id"]].append(r)
    for clip in map(Path, a.event_clips):
        by_clip.setdefault(clip.name, [])
    for cid, rows in sorted(by_clip.items()):
        clip = clips_dir / cid
        if not clip.exists():
            print(f"[{cid}] 사건 GT: clip 폴더 없음 -> 건너뜀")
            continue
        zones = {z["id"] for z in (lambda m: m.get("aisles", []) + m.get("no_parking", []))(
            load_map(data / "lots" / load_clip_meta(clip)["lot"]))}
        probs = [f"{r['event_id']}: " + ", ".join(x for x in [
            "종류" if r["type"] not in ("aisle_parking", "pass_through") else "",
            "구역" if r["zone_id"] not in zones else "",
            "시각" if not float(r["start_sec"]) < float(r["end_sec"]) else ""] if x) for r in rows]
        probs = [p for p in probs if not p.endswith(": ")]
        put = _put(rows, EVENT_HEADER, clip / "gt_events.csv", a.force)
        print(f"[{cid}] 사건 {len(rows)}개 -> {'gt_events.csv' if put else '(안 넣음)'} | "
              + ("검사 통과" if not probs else "문제 (종류/구역/시각): " + "; ".join(probs)))
    print(f"칸 GT 반영 {n_ok}개 파일 (검사 통과). 다음: python eval/evaluate.py --clips data/clips/lotA_cam[4-7]_* --split test")


if __name__ == "__main__":
    main()
