"""2차 footprint: 합성 데이터 (맵 좌표에서 직접)."""
import numpy as np
from shapely.geometry import Polygon

from parking.footprint import bottom_chain_img, estimate_footprint, footprint_states, slot_axes, slot_overlaps
from parking.map_builder import slot_row
from parking.occupancy import ENCROACH, FREE, OCCUPIED, slots_from_map

SLOTS = slots_from_map({"slots": slot_row([0, 0], [7.5, 0], [0, 5], 3, "A")})  # A01~A03, 2.5 x 5
U, V = slot_axes(SLOTS["A01"])
CAR = (1.8, 4.7)


def _visible_edges(x0, y0):
    """카메라가 (+x, -y) 쪽 아래에 있을 때 보이는 바닥 경계: 차 앞면(y0)과 오른쪽 옆면(x0+1.8)."""
    return np.array([[x0, y0], [x0 + 1.8, y0], [x0 + 1.8, y0 + 4.7]])


def test_bottom_chain_keeps_lower_side():
    poly = [[[0, 0], [100, 0], [100, 50], [60, 80], [0, 50]]]  # 아래쪽(y 큰 쪽)이 바닥
    chain = bottom_chain_img(poly)
    assert {tuple(p) for p in chain} == {(0, 50), (60, 80), (100, 50)}


def test_footprint_recovers_car_from_visible_edges():
    away = np.array([-0.6, 0.8])  # 카메라 반대쪽 = -x, +y
    fp = estimate_footprint(_visible_edges(0.35, 0.15), away, U, V, CAR)
    assert fp.symmetric_difference(Polygon([[0.35, 0.15], [2.15, 0.15], [2.15, 4.85], [0.35, 4.85]])).area < 1e-6


def test_centered_in_one_slot_is_occupied():
    fp = estimate_footprint(_visible_edges(2.85, 0.15), np.array([-0.6, 0.8]), U, V, CAR)
    ov = slot_overlaps(fp, SLOTS)
    assert ov["A02"] > 0.99
    assert footprint_states([ov], SLOTS, 0.15, 0.15) == {"A01": FREE, "A02": OCCUPIED, "A03": FREE}


def test_straddling_car_encroaches_both():
    # 차가 A01/A02 경계(x=2.5)를 0.8m 넘음: A02 쪽 1.0/1.8 = 44%, A01 쪽 56%
    fp = estimate_footprint(_visible_edges(1.5, 0.15), np.array([-0.6, 0.8]), U, V, CAR)
    ov = slot_overlaps(fp, SLOTS)
    assert abs(ov["A01"] - 1.0 / 1.8) < 1e-6
    assert footprint_states([ov], SLOTS, 0.15, 0.15) == {"A01": ENCROACH, "A02": ENCROACH, "A03": FREE}


def test_small_overhang_is_not_encroach():
    # 경계를 0.2m 넘음 = 11% < 15% -> 점유만
    fp = estimate_footprint(_visible_edges(0.9, 0.15), np.array([-0.6, 0.8]), U, V, CAR)
    st = footprint_states([slot_overlaps(fp, SLOTS)], SLOTS, 0.15, 0.15)
    assert st == {"A01": OCCUPIED, "A02": FREE, "A03": FREE}


def test_looking_straight_down_axis_centers():
    # 시선이 칸 길이 방향과 나란하면(옆면이 안 보임) 폭 방향은 보이는 앞면의 가운데 기준
    front = np.array([[0.5, 0.15], [2.0, 0.15]])
    fp = estimate_footprint(front, np.array([0.0, 1.0]), U, V, CAR)
    xs = np.asarray(fp.exterior.coords)[:, 0]
    assert np.isclose(xs.min(), 0.35) and np.isclose(xs.max(), 2.15)


def test_line_stepping_car_occupies_both_below_straddle():
    # 44%/56%로 걸친 차: 침범 기준(0.5) 미만이어도 15% 이상 걸친 칸은 둘 다 점유 (빈칸으로 안내하지 않음)
    fp = estimate_footprint(_visible_edges(1.5, 0.15), np.array([-0.6, 0.8]), U, V, CAR)
    st = footprint_states([slot_overlaps(fp, SLOTS)], SLOTS, 0.15, 0.5)
    assert st == {"A01": OCCUPIED, "A02": OCCUPIED, "A03": FREE}


def test_crosswise_car_in_aisle_lies_along_row():
    # 통로(y 5~11)를 가로지르는 차: 보이는 앞면 아래 경계가 x 방향 4.6m -> 사각형을 눕혀 칸을 덮지 않음
    from parking.footprint import looks_crosswise
    chain = np.array([[1.0, 5.4], [5.6, 5.4], [5.6, 7.2]])
    assert looks_crosswise(chain, U, V, 3.0, 1.5)
    away = np.array([0.0, 1.0])
    upright = estimate_footprint(chain, away, U, V, CAR)
    lying = estimate_footprint(chain, away, U, V, CAR[::-1])
    assert max(slot_overlaps(upright, SLOTS).values(), default=0) == 0  # (칸은 y 0~5라 둘 다 안 겹침: 방향만 확인)
    xs, ys = np.asarray(lying.exterior.coords).T
    assert np.isclose(np.ptp(xs), 4.7) and np.isclose(np.ptp(ys), 1.8)
    assert not looks_crosswise(np.array([[0.4, 0.2], [2.2, 0.2], [2.2, 4.9]]), U, V, 3.0, 1.5)  # 칸에 바르게 선 차
