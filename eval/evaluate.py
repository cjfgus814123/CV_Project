"""지표 계산 (CLAUDE.md 8): 혼동행렬, 빈칸 Precision, Recall, F1, Accuracy, 빈자리 수 MAE, 줄·조건별 표.

    python eval/evaluate.py --clips data/clips/lotA_cam3_mid
    python eval/evaluate.py --clips data/clips/lotA_* --split test --pred occupancy.csv

1차 기준: 상태 1(점유)·2(침범) = 점유, 0 = 빈칸. GT가 -1(판단불가)인 칸은 제외.
예측이 -1인데 GT는 판정 가능한 칸은 'pred_unknown'으로 따로 센다 (정확도 계산에선 제외, 커버리지로 보고).
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from parking.io import load_clip_meta, load_map


def load_states(path: Path) -> dict[int, dict[str, int]]:
    out: dict[int, dict[str, int]] = defaultdict(dict)
    for r in csv.DictReader(Path(path).open()):
        out[int(r["frame_idx"])][r["slot_id"]] = int(r["state"])
    return out


def _occ(s: int) -> bool:
    return s in (1, 2)


def score(pairs: list[tuple[int, int]]) -> dict:
    """(gt, pred) 쌍 목록 -> 지표. 둘 다 -1이 아닌 쌍만 혼동행렬에 들어간다."""
    cm = {"free_free": 0, "free_occ": 0, "occ_free": 0, "occ_occ": 0}  # gt_pred
    n_unknown = 0
    enc = {"enc_tp": 0, "enc_fp": 0, "enc_fn": 0}  # 2차: 침범(2) 칸 단위
    for g, p in pairs:
        if g == -1:
            continue
        if p == -1:
            n_unknown += 1
            continue
        if g == 2 or p == 2:
            enc["enc_tp" if g == p else ("enc_fp" if p == 2 else "enc_fn")] += 1
        cm[f"{'occ' if _occ(g) else 'free'}_{'occ' if _occ(p) else 'free'}"] += 1
    tp_f, fp_f, fn_f, tp_o = cm["free_free"], cm["occ_free"], cm["free_occ"], cm["occ_occ"]
    div = lambda a, b: a / b if b else float("nan")
    prec_f, rec_f = div(tp_f, tp_f + fp_f), div(tp_f, tp_f + fn_f)
    n = sum(cm.values())
    return {"n": n, **cm, **enc, "pred_unknown": n_unknown,
            "enc_precision": div(enc["enc_tp"], enc["enc_tp"] + enc["enc_fp"]),
            "enc_recall": div(enc["enc_tp"], enc["enc_tp"] + enc["enc_fn"]),
            "free_precision": prec_f, "free_recall": rec_f,
            "free_f1": div(2 * prec_f * rec_f, prec_f + rec_f),
            "occ_precision": div(tp_o, tp_o + fn_f), "occ_recall": div(tp_o, tp_o + fp_f),
            "accuracy": div(tp_f + tp_o, n)}


def evaluate_clip(gt: dict, pred: dict, rows: dict[str, str]) -> dict:
    """한 clip: 전체·줄별 지표 + 빈자리 수 MAE (GT 프레임 중 예측이 있는 프레임만).

    침범 장면 단위: GT에 침범(2) 칸이 있는 프레임에서 예측 침범 칸이 GT 침범 칸과 하나라도 겹치면 맞음.
    GT 침범 칸과 겹치지 않는 예측 침범이 있는 프레임은 오경보. (걸친 차의 두 칸을 다 2로 매기든
    넘어간 칸만 2로 매기든 같은 점수가 나오도록)
    """
    frames = [f for f in gt if f in pred]
    pairs, by_row, abs_err = [], defaultdict(list), []
    ev = {"ev_tp": 0, "ev_fn": 0, "ev_fp": 0}
    for f in frames:
        g2 = {s for s, v in gt[f].items() if v == 2}
        p2 = {s for s, v in pred[f].items() if v == 2 and gt[f].get(s, -1) != -1}
        if g2:
            ev["ev_tp" if p2 & g2 else "ev_fn"] += 1
        if p2 - g2 and not (p2 & g2):
            ev["ev_fp"] += 1
        g_free = p_free = 0
        for sid, g in gt[f].items():
            p = pred[f].get(sid, -1)
            pairs.append((g, p))
            by_row[rows.get(sid, "?")].append((g, p))
            if g != -1:
                g_free += g == 0
                p_free += p == 0
        abs_err.append(abs(g_free - p_free))
    res = {**score(pairs), **ev}
    res["frames"] = len(frames)
    res["free_count_mae"] = sum(abs_err) / len(abs_err) if abs_err else float("nan")
    res["by_row"] = {r: score(v) for r, v in sorted(by_row.items())}
    return res, pairs, abs_err


def evaluate(clips: list[Path], pred_name: str, split: str | None, data: Path, out_root: Path,
             variant_dir: str | None = None) -> dict:
    """여러 clip 합산 + clip별·조건별 지표. variant_dir가 있으면 outputs/<clip>/<variant_dir>/<pred_name>을 읽는다."""
    all_pairs, all_err, per_clip, by_cond = [], [], {}, defaultdict(list)
    ev = {"ev_tp": 0, "ev_fn": 0, "ev_fp": 0}
    for clip in clips:
        meta = load_clip_meta(clip)
        if split and meta.get("split") != split:
            continue
        gt_path = clip / "gt_occupancy.csv"
        pred_path = out_root / clip.name / (variant_dir or "") / pred_name
        if not gt_path.exists() or not pred_path.exists():
            print(f"건너뜀 {clip.name}: {'GT' if not gt_path.exists() else '예측'} 없음")
            continue
        rows = {s["id"]: s.get("row", "?") for s in load_map(data / "lots" / meta["lot"])["slots"]}
        res, pairs, err = evaluate_clip(load_states(gt_path), load_states(pred_path), rows)
        per_clip[clip.name] = res
        for k in ev:
            ev[k] += res[k]
        all_pairs += pairs
        all_err += err
        by_cond[meta.get("condition", "?")] += pairs
    total = {**score(all_pairs), **ev}
    total["free_count_mae"] = sum(all_err) / len(all_err) if all_err else float("nan")
    total["frames"] = len(all_err)
    return {"total": total, "by_condition": {c: score(v) for c, v in by_cond.items()}, "per_clip": per_clip}


def fmt(r: dict) -> str:
    return (f"n={r['n']:5d}  빈칸P {r['free_precision']:.3f}  빈칸R {r['free_recall']:.3f}  빈칸F1 {r['free_f1']:.3f}  "
            f"점유R {r['occ_recall']:.3f}  Acc {r['accuracy']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--split", choices=["tune", "test"], default=None, help="meta.json의 split으로 거르기 (없으면 전부)")
    ap.add_argument("--pred", default="occupancy.csv")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--json", default=None, help="결과를 JSON으로 저장할 경로")
    a = ap.parse_args()

    res = evaluate([Path(c) for c in a.clips], a.pred, a.split, Path(a.data), Path(a.out))
    t = res["total"]
    if not t["n"]:
        raise SystemExit("평가할 GT가 없음 (gt_occupancy.csv 확인)")
    print(f"[전체] {t['frames']}프레임  {fmt(t)}  빈자리수 MAE {t['free_count_mae']:.2f}  (예측 판단불가 {t['pred_unknown']})")
    print(f"  혼동행렬 (GT→예측): 빈→빈 {t['free_free']}  빈→점유 {t['free_occ']}  "
          f"점유→빈 {t['occ_free']} (치명적)  점유→점유 {t['occ_occ']}")
    print(f"  2차 침범: Precision {t['enc_precision']:.3f}  Recall {t['enc_recall']:.3f}  "
          f"(맞음 {t['enc_tp']}, 오경보 {t['enc_fp']}, 놓침 {t['enc_fn']})")
    print(f"  2차 침범 장면 단위: 맞음 {t['ev_tp']}  놓침 {t['ev_fn']}  오경보 {t['ev_fp']}")
    for c, r in res["by_condition"].items():
        print(f"[조건 {c}] {fmt(r)}")
    for name, r in res["per_clip"].items():
        print(f"[{name}] {fmt(r)}  MAE {r['free_count_mae']:.2f}")
        for row, rr in r["by_row"].items():
            print(f"    줄 {row}: {fmt(rr)}")
    if a.json:
        Path(a.json).write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
