import cv2
import numpy as np
import pytest
from app.domain.logic.layout_checker import check_passport_layout

@pytest.fixture
def no_face_image():
    return np.zeros((900,1200,3), dtype=np.uint8)

@pytest.fixture
def face_image(tmp_path):
    # draw a simple face-like circle on left side
    img = np.zeros((900,1200,3), dtype=np.uint8)
    cv2.circle(img, (200,300), 100, (255,255,255), -1)
    return img

def test_no_face(no_face_image):
    ok, msg = check_passport_layout(no_face_image)
    assert not ok
    assert "No face detected" in msg

def test_face_position_correct(face_image):
    ok, msg = check_passport_layout(face_image)
    assert ok
    assert msg == "Basic layout OK"