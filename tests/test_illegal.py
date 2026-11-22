"""3차 정지 시간 판정, 이벤트 지표: 합성 궤적."""
import sys
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

from parking.illegal import StopDetector
from parking.map_builder import aisle_between, slot_row

sys.path.insert(0, str(Path(__file__).parents[1] / "eval"))
from evaluate import score_events  # noqa: E402

AISLE = {"AISLE1": Polygon([[0, 5], [30, 5], [30, 11], [0, 11]])}


def _run(traj, fps=2.0):
    """traj(t) -> 맵 좌표. 0~200초를 fps로 흘려 이벤트 목록."""
    det = StopDetector(AISLE, v_stop=0.3, t_stop=60, window_sec=5)
    for k in range(int(200 * fps)):
        t = k / fps
        p = traj(t)
        det.update(t, [] if p is None else [(1, np.array(p) + np.random.default_rng(k).normal(0, 0.1, 2))])
    return det.finish()


def test_long_stop_in_aisle_is_event():
    # 0~30초 1m/s로 들어옴, 30~130초 통로에 정지(흔들림 0.1m), 이후 떠남
    ev = _run(lambda t: (t - 24.0, 8) if t < 30 else ((6, 8) if t < 130 else (6 + (t - 130) * 3, 8)))
    assert len(ev) == 1
    e = ev[0]
    assert e["zone_id"] == "AISLE1" and 25 <= e["start_sec"] <= 36 and 85 <= e["detect_sec"] <= 96
    assert 128 <= e["end_sec"] <= 134


def test_pass_through_and_short_stop_are_not_events():
    assert _run(lambda t: (t * 1.0 % 30, 8)) == []  # 1m/s로 통로를 계속 지나감
    assert _run(lambda t: (5, 8) if 50 <= t < 90 else (min(t, 199) * 2 % 30, 8)) == []  # 40초 정지 < 60


def test_stop_in_slot_is_not_event():
    assert _run(lambda t: (6, 2)) == []  # 칸(통로 밖)에 계속 서 있음


def test_aisle_between_rows():
    a = slot_row([0, 0], [5, 0], [0, 5], 2, "A")
    b = slot_row([0, 11], [5, 11], [0, 16], 2, "B")
    poly = Polygon(aisle_between(a, b, "X")["polygon"])
    assert poly.is_valid and abs(poly.area - 5 * 6) < 1e-6


def test_score_events():
    gt = [{"type": "aisle_parking", "zone_id": "A", "start_sec": "40", "end_sec": "130"},
          {"type": "pass_through", "zone_id": "A", "start_sec": "150", "end_sec": "158"}]
    pred = [{"zone_id": "A", "start_sec": "38", "end_sec": "131", "detect_sec": "98"},
            {"zone_id": "A", "start_sec": "150", "end_sec": "215", "detect_sec": "210"}]
    s = score_events(gt, pred)
    assert s["tp"] == 1 and s["false_alarms"] == 1 and s["pass_false_alarms"] == 1
    assert s["precision"] == 0.5 and s["recall"] == 1.0 and s["delay_mean_sec"] == 58.0
