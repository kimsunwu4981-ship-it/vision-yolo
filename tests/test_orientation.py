import numpy as np
import pytest
from app.orientation import orient_bgr
from app.geometry import transform_boxes


def test_display_matrix_clockwise_pixels_and_bbox():
    image=np.repeat(np.array([[1,2,3],[4,5,6]],np.uint8)[:,:,None],3,axis=2)
    actual,meta=orient_bgr(image,-90)
    assert actual[:,:,0].tolist()==[[4,1],[5,2],[6,3]]
    assert transform_boxes([[0,0,1,1]],meta['encoded_to_display_matrix']).tolist()==[[1,0,2,1]]
    assert meta['encoded_source_width']==3 and meta['encoded_source_height']==2


@pytest.mark.parametrize('angle',[-90,0,90,180,270])
def test_rotation_box_roundtrip_and_extent(angle):
    image=np.zeros((720,1280,3),np.uint8)
    pixels,meta=orient_bgr(image,angle)
    box=np.array([[20.,30.,180.,290.]])
    mapped=transform_boxes(box,meta['encoded_to_display_matrix'])
    np.testing.assert_allclose(transform_boxes(mapped,meta['display_to_encoded_matrix']),box)
    bounds=transform_boxes([[0,0,1280,720]],meta['encoded_to_display_matrix'])
    assert bounds.tolist()==[[0,0,pixels.shape[1],pixels.shape[0]]]


def test_non_right_angle_rejected():
    with pytest.raises(ValueError):orient_bgr(np.zeros((3,4,3),np.uint8),17)
