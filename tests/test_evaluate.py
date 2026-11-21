import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "eval"))
sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
from evaluate import score  # noqa: E402
from sample_frames import pick  # noqa: E402


def test_score_counts_and_skips_unknown():
    # (gt, pred): 빈→빈 2, 점유→빈 1(치명), 빈→점유 1, 침범(2)→점유 1, GT -1 제외, 예측 -1은 따로
    r = score([(0, 0), (0, 0), (1, 0), (0, 1), (2, 1), (-1, 0), (1, -1)])
    assert (r["free_free"], r["occ_free"], r["free_occ"], r["occ_occ"]) == (2, 1, 1, 1)
    assert r["n"] == 5 and r["pred_unknown"] == 1
    assert math.isclose(r["free_precision"], 2 / 3) and math.isclose(r["free_recall"], 2 / 3)
    assert math.isclose(r["accuracy"], 3 / 5)


def test_pick_even_spacing_and_cap():
    times = [i * 0.5 for i in range(201)]  # 0~100초, 2fps
    assert [times[k] for k in pick(times, 10, 100)] == [float(t) for t in range(0, 101, 10)]
    assert len(pick(times, 1, 20)) == 20
