import cv2
import numpy as np
import pytest
from app.domain.logic.mrz_extractor import validate_mrz

class DummyMRZ:
    def __init__(self, **kwargs):
        self.country = 'USA'
        self.number = 'X123456'
        self.date_of_birth = '1980-01-01'
        self.expiration_date = '2030-12-31'
        self.names = 'JOHN'
        self.surname = 'DOE'
        self.sex = 'M'
        self.nationality = 'USA'
        self.valid_score = 80
        self.valid_number = True
        self.valid_date_of_birth = True
        self.valid_expiration_date = True
        self.valid_composite = True

@pytest.fixture(autouse=True)
def patch_read_mrz(monkeypatch, tmp_path):
    # patch read_mrz to return DummyMRZ
    def fake_read_mrz(path):
        return DummyMRZ()
    import passporteye
    monkeypatch.setattr('passporteye.read_mrz', fake_read_mrz)
    yield

@pytest.fixture
def sample_image():
    return np.zeros((900,1200,3), dtype=np.uint8)


def test_validate_mrz_success(sample_image):
    ok, data = validate_mrz(sample_image)
    assert ok
    assert data['country'] == 'USA'
    assert data['validation_details']['valid_score'] == 80

def test_validate_mrz_failure(monkeypatch, sample_image):
    # patch to return None
    monkeypatch.setattr('passporteye.read_mrz', lambda p: None)
    ok, data = validate_mrz(sample_image)
    assert not ok
    assert data is None