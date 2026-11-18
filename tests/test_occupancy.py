import numpy as np
import pytest
from shapely.geometry import box

from conftest import slot_rows
from parking.occupancy import (OCCUPIED, FREE, UNKNOWN, apply_visibility, assign_slots,
                               slot_visibility, slots_from_map, temporal_smooth)


def test_assign_inside_boundary_and_aisle():
    slots = slot_rows()
    pts = np.array([[-3.75, 12.5],   # A01 안
                    [1.25, 9.8],     # A03 앞 경계에서 0.2m 밖 -> tol 0.3 안이라 A03
                    [3.75, 8.0]])    # 통로 (2m 밖) -> 무시
    s = assign_slots(pts, slots, tol_m=0.3)
    assert s["A01"] == OCCUPIED and s["A03"] == OCCUPIED
    assert s["A04"] == FREE and sum(v == OCCUPIED for v in s.values()) == 2


def test_assign_on_shared_edge_picks_closer_centroid():
    s = assign_slots(np.array([[0.0, 11.0]]), slot_rows(), tol_m=0.3)  # A02|A03 경계, 둘 다 거리 0
    assert sum(v == OCCUPIED for v in s.values()) == 1


def test_assign_no_cars():
    assert set(assign_slots(np.zeros((0, 2)), slot_rows(), 0.3).values()) == {FREE}


def test_visibility_and_unknown():
    slots = {"S1": box(0, 0, 2, 5), "S2": box(2, 0, 4, 5), "S3": box(10, 0, 12, 5)}
    vis = slot_visibility(slots, box(-1, -1, 3, 6))  # S1 전부, S2 절반, S3 안 보임
    assert vis == pytest.approx({"S1": 1.0, "S2": 0.5, "S3": 0.0})
    st = apply_visibility({"S1": OCCUPIED, "S2": FREE, "S3": OCCUPIED}, vis, vis_unknown=0.3)
    assert st == {"S1": OCCUPIED, "S2": FREE, "S3": UNKNOWN}


def test_temporal_smooth():
    h = [{"A": 1}, {"A": 1}, {"A": 0}, {"A": 1}, {"A": 1}]
    assert temporal_smooth(h, 5) == {"A": 1}                  # 한 프레임 깜빡임 제거
    assert temporal_smooth([{"A": 0}, {"A": 1}], 5) == {"A": 1}  # 기록 부족 + 동점 -> 점유
    assert temporal_smooth(h + [{"A": 0}] * 3, 5) == {"A": 0}   # 실제로 빠지면 따라감


def test_slots_from_map():
    m = {"slots": [{"id": "A01", "row": "A", "polygon": [[0, 0], [2.5, 0], [2.5, 5], [0, 5]]}]}
    assert slots_from_map(m)["A01"].area == pytest.approx(12.5)
