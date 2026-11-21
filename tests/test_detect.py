import numpy as np

from parking.detect import mask_to_polys, polys_to_mask


def test_mask_polys_round_trip_two_pieces():
    """가로등에 잘려 두 조각이 된 마스크 -> 다각형 2개 -> 다시 마스크 (이어 붙이지 않음)."""
    m = np.zeros((100, 200), bool)
    m[20:80, 20:90] = True
    m[20:80, 110:180] = True
    polys = mask_to_polys(m)
    assert len(polys) == 2
    back = polys_to_mask(polys, (200, 100))
    assert not back[50, 100]  # 두 조각 사이 틈은 비어 있어야 함
    assert (back == m).mean() > 0.99


def test_dedup_drops_box_inside_box():
    from parking.detect import dedup

    big = {"conf": 0.6, "bbox_img": [0, 0, 100, 100]}
    inner = {"conf": 0.54, "bbox_img": [10, 10, 40, 50]}  # IoU 0.12지만 작은 박스는 완전히 안에 있음
    other = {"conf": 0.9, "bbox_img": [90, 0, 190, 100]}  # 옆 차: 살짝만 겹침
    assert dedup([inner, big, other], 0.8) == [other, big]
