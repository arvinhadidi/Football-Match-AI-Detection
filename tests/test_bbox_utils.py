from ..utils.bbox_utils import get_center_of_bbox

def test_get_center_of_bbox():
    bbox = [10, 20, 30, 40]
    center_x, center_y = get_center_of_bbox(bbox)
    assert center_x == 20
    assert center_y == 30
