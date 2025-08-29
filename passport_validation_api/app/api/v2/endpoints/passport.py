from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from app.api.v2.deps import get_tesseract_cmd
import logging
import base64
import cv2
import numpy as np

from app.services.orchestrator import PassportValidationOrchestrator
from app.schemas.response import PassportValidationResponse
from app.utils.image_io import load_image_from_upload
from app.domain.logic.mrz_extractor import MRZExtractor

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/validate", response_model=PassportValidationResponse)
async def validate_passport(
    file: UploadFile = File(...),
    include_debug: bool = Query(False, description="Include debug information in response")
):
    """
    Validate a passport image by checking quality, layout, and extracting MRZ data.
    
    Args:
        file: Uploaded passport image file
        include_debug: Whether to include debug information
        
    Returns:
        Validation results including MRZ data if successful
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400, 
                detail="File must be an image"
            )
        
        # Load image
        image = await load_image_from_upload(file)
        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Could not process uploaded image"
            )
        
        # Initialize orchestrator and run validation
        orchestrator = PassportValidationOrchestrator()
        result = await orchestrator.validate_passport(
            image=image,
            include_debug=include_debug
        )
        
        # Return appropriate response
        if result['is_valid']:
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": "Passport validation successful",
                    "data": result
                }
            )
        else:
            return JSONResponse(
                status_code=422,
                content={
                    "success": False,
                    "message": "Passport validation failed",
                    "errors": result['errors'],
                    "data": result if include_debug else None
                }
            )
            
    except Exception as e:
        logger.error(f"Passport validation endpoint error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during validation: {str(e)}"
        )

@router.post("/extract-mrz-image")
async def extract_mrz_image(
    file: UploadFile = File(...),
    use_ocr: bool = Query(True, description="Use OCR for MRZ validation"),
    debug: bool = Query(False, description="Return debug information")
):
    """
    Extract only the MRZ image region base64 from passport image (for testing/debugging).
    """
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        image = await load_image_from_upload(file)
        if image is None:
            raise HTTPException(status_code=400, detail="Could not process uploaded image")
        
        # Extract MRZ region only
        mrz_extractor = MRZExtractor()
        mrz_image, metadata = mrz_extractor.extract_mrz_region(
            image, 
            use_ocr=use_ocr, 
            debug=debug
        )
        
        # Convert NumPy array to base64 string for JSON serialization
        mrz_image_base64 = None
        if mrz_image is not None:
            # Convert to PNG format
            _, buffer = cv2.imencode('.png', mrz_image)
            # Convert to base64 string
            mrz_image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Clean metadata to ensure it's JSON serializable
        # Use the orchestrator's helper method for more robust cleaning
        orchestrator = PassportValidationOrchestrator()
        clean_metadata = orchestrator._clean_numpy_from_dict(metadata)
        
        return JSONResponse(
            status_code=200,
            content={
                "success": mrz_image is not None,
                "metadata": clean_metadata,
                "mrz_image_base64": mrz_image_base64,
                "message": "MRZ extracted successfully" if mrz_image is not None else "MRZ not detected"
            }
        )
        
    except Exception as e:
        logger.error(f"MRZ extraction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/extract-mrz-data")
async def extract_mrz_data(
    file: UploadFile = File(...),
    use_ocr: bool = Query(True, description="Use OCR for MRZ validation"),
    debug: bool = Query(False, description="Return debug information")
):
    """
    Extract only the MRZ region from passport image (for testing/debugging).
    Returns parsed MRZ fields (ICAO 9303) in place of the MRZ image base64.
    """
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        image = await load_image_from_upload(file)
        if image is None:
            raise HTTPException(status_code=400, detail="Could not process uploaded image")
        
        # Extract MRZ region only
        mrz_extractor = MRZExtractor()
        mrz_image, metadata = mrz_extractor.extract_mrz_region(
            image, 
            use_ocr=use_ocr, 
            debug=debug
        )
        
        # Clean metadata to ensure it's JSON serializable
        orchestrator = PassportValidationOrchestrator()
        clean_metadata = orchestrator._clean_numpy_from_dict(metadata)
        
        # Parse MRZ into structured fields (if MRZ image was found)
        mrz_fields = None
        if mrz_image is not None:
            # Use orchestrator's async parser which performs OCR + ICAO parsing + check-digit validation
            mrz_parsed = await orchestrator._parse_mrz_data(mrz_image)
            if mrz_parsed:
                # MRZData is a Pydantic model in your orchestrator — convert to dict
                mrz_fields = mrz_parsed.dict()
            else:
                # Parsing failed despite MRZ region detected
                mrz_fields = None
        
        return JSONResponse(
            status_code=200,
            content={
                "success": mrz_fields is not None,
                "metadata": clean_metadata,
                # here we return the parsed MRZ fields (or null if not parsed)
                "mrz_fields": mrz_fields,
                "message": (
                    "MRZ parsed successfully" if mrz_fields is not None
                    else ("MRZ detected but parsing failed" if mrz_image is not None else "MRZ not detected")
                )
            }
        )
        
    except Exception as e:
        logger.error(f"MRZ extraction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
