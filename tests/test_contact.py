import numpy as np
import pytest

from conftest import IMG_H, IMG_W, car_mask_poly, slot_rows
from shapely.geometry import box

from parking.contact import contact_map, contact_point_img
from parking.detect import mask_to_polys
from parking.occupancy import nearest_slot

SIZE = (IMG_W, IMG_H)


def rect_mask(notch=False, split=False):
    """차 마스크 흉내: x 100~300, y 100~200. notch: 아래 가운데를 가로등이 파고듦, split: 완전히 둘로 가름."""
    m = np.zeros((IMG_H, IMG_W), bool)
    m[100:200, 100:300] = True
    if notch:
        m[140:200, 190:210] = False
    if split:
        m[100:200, 190:210] = False
    return mask_to_polys(m)


@pytest.mark.parametrize("kw", [{}, {"notch": True}, {"split": True}])
def test_mask_bottom_center_ignores_occluder(kw):
    pt, trunc = contact_point_img(rect_mask(**kw), [100, 100, 300, 200], "mask_bottom_center", SIZE, 2)
    assert pt == pytest.approx([199.5, 199], abs=1.0)  # 홈 위(140)로 튀지 않음
    assert not trunc


def test_bbox_modes_and_fallback():
    bb = [100, 100, 300, 200]
    assert contact_point_img(None, bb, "bbox_center", SIZE, 2)[0] == pytest.approx([200, 150])
    assert contact_point_img(None, bb, "bbox_bottom_center", SIZE, 2)[0] == pytest.approx([200, 200])
    assert contact_point_img([], bb, "mask_bottom_center", SIZE, 2)[0] == pytest.approx([200, 200])
    with pytest.raises(ValueError):
        contact_point_img(None, bb, "roof", SIZE, 2)


@pytest.mark.parametrize("bb,expected", [
    ([0, 100, 50, 200], True),               # 왼쪽 경계
    ([1800, 100, IMG_W - 1, 200], True),     # 오른쪽 경계
    ([100, 900, 300, IMG_H - 1], True),      # 아래 경계 (접점이 잘림)
    ([100, 0, 300, 200], False),             # 위쪽만 닿음 -> 접점과 무관
])
def test_truncated(bb, expected):
    assert contact_point_img(None, bb, "bbox_bottom_center", SIZE, 2)[1] is expected


def car_contacts(cam, center, size):
    """합성 차 -> 방식별 맵 접점 {mode: (x, y)}"""
    polys, bb = car_mask_poly(cam, center, size)
    pts = {m: contact_point_img(polys, bb, m, SIZE, 2)[0] for m in ("mask_bottom_center", "bbox_center")}
    return {
        "bbox_center": contact_map(pts["bbox_center"], cam["H"], 0.0),
        "mask_bottom": contact_map(pts["mask_bottom_center"], cam["H"], 0.0),
        "mask_bottom+0.9": contact_map(pts["mask_bottom_center"], cam["H"], 0.9),
    }


def test_sideways_car_bbox_center_lands_in_wrong_slot(cam):
    """옆면이 카메라를 향한 차(칸 깊이 2.5m): 높이 때문에 박스 중심은 뒤 칸으로 밀려나고,
    지면 접점은 맞는 칸에 찍힌다. (논문 ② 베이스라인이 틀리는 경우)"""
    slots = {"F": box(0, 12.5, 5, 15), "S": box(0, 15, 5, 17.5), "R": box(0, 17.5, 5, 20)}  # 앞/본/뒤 칸
    c = car_contacts(cam, (2.5, 16.25), (4.7, 1.8))
    assert nearest_slot(c["bbox_center"], slots, 0.3) == "R"
    assert nearest_slot(c["mask_bottom"], slots, 0.3) == "S"
    assert nearest_slot(c["mask_bottom+0.9"], slots, 0.3) == "S"
    assert c["mask_bottom+0.9"][1] == pytest.approx(16.25, abs=0.1)  # 차 중심에 거의 일치


@pytest.mark.parametrize("y", [12.5, 17.5])
def test_facing_car_offset_moves_contact_off_slot_edge(cam, y):
    """정면이 카메라를 향한 차: 마스크 아래 점은 칸 앞 경계에서 0.15m 안쪽에 붙는다 (H 오차에 취약).
    0.9m 오프셋은 경계에서 1m 이상 떨어뜨린다."""
    slots = slot_rows()
    c = car_contacts(cam, (1.25, y), (1.8, 4.7))
    front = y - 2.5
    assert c["mask_bottom"][1] - front == pytest.approx(0.15, abs=0.05)
    assert c["mask_bottom+0.9"][1] - front > 1.0
    assert nearest_slot(c["mask_bottom+0.9"], slots, 0.3) == nearest_slot(np.array([1.25, y]), slots, 0)
