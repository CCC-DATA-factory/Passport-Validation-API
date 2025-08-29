import numpy as np
import pytest
import asyncio
from fastapi import UploadFile
from io import BytesIO
from app.services.orchestrator import process_passport_image

class DummyFile:
    def __init__(self, content):
        self.file = BytesIO(content)
    async def read(self):
        return self.file.read()

@pytest.fixture(autouse=True)
def patch_helpers(monkeypatch):
    # patch quality_checker
    monkeypatch.setattr(
        'app.domain.logic.quality_checker.check_image_quality',
        lambda img: (True, 'ok')
    )
    # patch mrz_extractor
    monkeypatch.setattr(
        'app.domain.logic.mrz_extractor.validate_mrz',
        lambda img: (True, {'country':'X','passport_number':'Y','birth_date':'1980-01-01','expiry_date':'2030-01-01','name':'A','surname':'B','gender':'M','nationality':'X','validation_details':{}})
    )
    # patch layout_checker
    monkeypatch.setattr(
        'app.domain.logic.layout_checker.check_passport_layout',
        lambda img: (True, 'layout ok')
    )
    yield

@pytest.mark.asyncio
def test_orchestrator_success():
    # create a blank JPEG header
    content = b'\xff\xd8' + b'0'*100 + b'\xff\xd9'
    file = DummyFile(content)
    result = pytest.run(asyncio.ensure_future(process_passport_image(file)))
    assert result['valid'] is True
    assert 'data' in result
    assert 'checks' in result

@pytest.mark.asyncio
def test_orchestrator_quality_fail():
    # patch quality to fail
    import app.domain.logic.quality_checker as qc
    qc.check_image_quality = lambda img: (False, 'bad')
    content = b'\xff\xd8' + b'0'*100 + b'\xff\xd9'
    file = DummyFile(content)
    with pytest.raises(ValueError) as exc:
        result = pytest.run(asyncio.ensure_future(process_passport_image(file)))
    assert 'bad' in str(exc.value)
