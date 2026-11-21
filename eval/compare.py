"""베이스라인(논문 ②) vs 제안 방식, ablation 표 (CLAUDE.md 8).

    python eval/compare.py --clips data/clips/lotA_cam3_mid
    python eval/compare.py --clips data/clips/lotA_* --split test

같은 검출 캐시·같은 GT로 설정만 바꿔 파이프라인을 다시 돌린다 (영상 없이 CSV만, clip당 수십 초).
결과: outputs/<clip>/variants/<이름>/occupancy.csv, 표는 outputs/compare_<split>.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from parking.baseline import BASELINE, with_overrides
from parking.io import load_clip_meta, load_config
from parking.pipeline import run

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import evaluate  # noqa: E402

# 이름: (설명, 설정 덮어쓰기). 기본 설정 = 제안 방식 (configs/default.yaml)
VARIANTS = {
    "proposed": ("제안: 지면 접점(마스크 껍질 아래) + 맵 판정 + 다수결", {}),
    "proposed_no_smooth": ("제안, 시간 보정 없음", {"occupancy": {"smooth_n": 1}}),
    "proposed_offset_0.9": ("제안 + 접점 깊이 오프셋 0.9m", {"contact": {"depth_offset_m": 0.9}}),
    "bbox_bottom": ("접점 = 박스 아래변 중앙", {"contact": {"mode": "bbox_bottom_center"}}),
    "baseline_paper2": ("베이스라인(논문 ②): 박스 중심, 허용 거리 0, 시간 보정 없음", BASELINE),
    "baseline_smooth": ("베이스라인 + 다수결", {**BASELINE, "occupancy": {**BASELINE["occupancy"], "smooth_n": 5}}),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--split", choices=["tune", "test"], default=None)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()

    cfg, out_root, data = load_config(a.config), Path(a.out), Path(a.data)
    clips = [Path(c) for c in a.clips if (Path(c) / "gt_occupancy.csv").exists()
             and (a.split is None or load_clip_meta(c).get("split") == a.split)]
    if not clips:
        raise SystemExit("GT(gt_occupancy.csv)가 있는 clip이 없음")
    rows = []
    for name, (desc, ov) in VARIANTS.items():
        for clip in clips:
            run(clip, with_overrides(cfg, ov), out_root, data, video=False,
                out_dir=out_root / clip.name / "variants" / name, verbose=False)
        t = evaluate(clips, "occupancy.csv", a.split, data, out_root, f"variants/{name}")["total"]
        rows.append((name, desc, t))
        print(f"{name:22s} 빈칸P {t['free_precision']:.3f}  빈칸R {t['free_recall']:.3f}  "
              f"점유R {t['occ_recall']:.3f}  Acc {t['accuracy']:.3f}  MAE {t['free_count_mae']:.2f}  "
              f"점유→빈(치명) {t['occ_free']}")

    md = [f"# 방식 비교 (split={a.split or 'all'}, clips: {', '.join(c.name for c in clips)})", "",
          f"GT {rows[0][2]['frames']}프레임, 칸 판정 {rows[0][2]['n']}건. 설정: {a.config}", "",
          "| 방식 | 설명 | 빈칸 Precision | 빈칸 Recall | 빈칸 F1 | 점유 Recall | Accuracy | 빈자리 수 MAE | 점유→빈칸 (치명) |",
          "|---|---|---|---|---|---|---|---|---|"]
    md += [f"| {n} | {d} | {t['free_precision']:.3f} | {t['free_recall']:.3f} | {t['free_f1']:.3f} | "
           f"{t['occ_recall']:.3f} | {t['accuracy']:.3f} | {t['free_count_mae']:.2f} | {t['occ_free']} |" for n, d, t in rows]
    path = out_root / f"compare_{a.split or 'all'}.md"
    path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
