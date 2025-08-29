from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from app.services.orchestrator import process_passport_image
from app.api.v1.deps import get_tesseract_cmd
import logging
import base64
import cv2
import numpy as np
from app.utils.image_io import load_image_from_upload
from app.domain.logic.mrz_extractor import MRZExtractor

logger = logging.getLogger(__name__)

router = APIRouter()

#--------------------------------------------------V1----------------------------------------------------------

@router.post("/validate-passport") 
async def validate_passport(
    file: UploadFile = File(...), 
    tesseract_cmd: str = Depends(get_tesseract_cmd)  
):
    """
    Endpoint to validate a passport image and extract MRZ data using PassportEye.
    
    Parameters:
    - file: The uploaded passport image (required)
    - tesseract_cmd: Path to the Tesseract executable, injected via FastAPI Depends
    
    Returns:
    - JSON response containing passport extraction results
    - 400 error if input or processing is invalid
    - 500 error for unexpected internal failures
    """
    try:
        result = await process_passport_image(file)
        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:

        raise HTTPException(status_code=500, detail=f"Internal error: {e}")




