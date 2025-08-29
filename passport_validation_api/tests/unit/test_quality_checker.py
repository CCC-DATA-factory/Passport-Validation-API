import cv2
import numpy as np
import pytest
from app.domain.logic.quality_checker import check_image_quality

@pytest.fixture
def blank_image():
    # create a blank low-res image
    return np.zeros((500, 400, 3), dtype=np.uint8)

@pytest.fixture
def sharp_image(tmp_path):
    # create a high-res but blurry image vs a sharp one
    img = np.full((900, 1200, 3), 255, dtype=np.uint8)
    # draw sharp edges
    cv2.line(img, (0,0), (1200,900), (0,0,0), 5)
    return img

def test_low_resolution(blank_image):
    ok, msg = check_image_quality(blank_image)
    assert not ok
    assert "resolution too low" in msg

def test_blurry_image(tmp_path):
    # blurred high-res
    img = cv2.GaussianBlur(sharp_image(tmp_path), (51,51), 0)
    ok, msg = check_image_quality(img)
    assert not ok
    assert "blurry" in msg

def test_good_image(sharp_image):
    ok, msg = check_image_quality(sharp_image)
    assert ok
    assert msg == "Image quality OK"