import cv2
from passporteye import read_mrz
from typing import Optional, Tuple, Dict
from datetime import datetime, date
import numpy as np
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import logging
import re
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logging.warning("Tesseract not available. OCR validation will be disabled.")

logger = logging.getLogger(__name__)

def _adjust_century(two_digit_year: int, reference_year: int) -> int:
    """
    Adjust a two-digit year to a full year based on a reference year.
    If the two-digit year <= current year's two-digit => 2000+; else 1900+.
    """
    yy = two_digit_year
    cutoff = reference_year % 100
    if yy <= cutoff:
        return 2000 + yy
    return 1900 + yy


def _parse_mrz_date(raw: str) -> date:
    """
    Parse a YYMMDD string from MRZ into a proper date with century correction.
    """
    yy = int(raw[0:2])
    mm = int(raw[2:4])
    dd = int(raw[4:6])
    full_year = _adjust_century(yy, datetime.today().year)
    return date(full_year, mm, dd)


def validate_mrz(image) -> Tuple[bool, Optional[Dict]]:
    # Save temp file for PassportEye
    temp_path = "/tmp/passport.jpg"
    cv2.imwrite(temp_path, image)

    mrz = read_mrz(temp_path)
    if mrz is None or not hasattr(mrz, 'valid_score') or mrz.valid_score <= 50:
        return False, None

    # Parse MRZ dates with correct century handling
    dob = _parse_mrz_date(mrz.date_of_birth)
    exp = _parse_mrz_date(mrz.expiration_date)

    return True, {
        "country": mrz.country,
        "passport_number": mrz.number,
        "birth_date": dob,
        "expiry_date": exp,
        "name": mrz.names,
        "surname": mrz.surname,
        "gender": mrz.sex,
        "nationality": mrz.nationality,
        "validation_details": {
            "valid_score": mrz.valid_score,
            "number_valid": mrz.valid_number,
            "dob_valid": mrz.valid_date_of_birth,
            "expiry_valid": mrz.valid_expiration_date,
            "composite_valid": mrz.valid_composite
        }
    }


#--------------------------------------------------V2 --------------------------------------------------------------------

class MRZExtractor:
    """Handles MRZ detection and extraction from passport images."""
    
    MRZ_REGEX = re.compile(r'^[A-Z0-9<]{10,44}$')
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize MRZ extractor with configuration."""
        self.config = config or {}
        self.resize_width = self.config.get('resize_width', 1000)
        self.expand_pixels = self.config.get('expand_pixels', 10)
        self.vertical_merge_gap_factor = self.config.get('vertical_merge_gap_factor', 1.8)
        
    def extract_mrz_region(
    self, 
    image: np.ndarray, 
    use_ocr: bool = False,
    debug: bool = False
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Extract MRZ region from passport image.
        
        Args:
            image: Input BGR image array
            use_ocr: Whether to use OCR for validation
            debug: Whether to return debug information
            
        Returns:
            Tuple of (cropped_mrz_image, metadata)
            - cropped_mrz_image: Cropped and deskewed MRZ region or None if not found
            - metadata: Dictionary containing extraction results and debug info
        """
        try:
            result = self._detect_and_crop_mrz(
                image, 
                use_ocr=use_ocr, 
                debug=debug
            )
            
            # Handle the result properly regardless of debug mode
            warped = None
            debug_info = {}
            
            if isinstance(result, tuple):
                # Result is a tuple - could be (warped, debug_info) or just (warped,)
                if len(result) >= 1:
                    warped = result[0]
                if len(result) >= 2:
                    debug_info = result[1] or {}
            else:
                # Result is just the warped image directly
                warped = result
            
            # Build metadata
            metadata = {
                'success': warped is not None,
                'ocr_available': TESSERACT_AVAILABLE,
                'mrz_detected': warped is not None
            }
            
            # Add debug info if available and requested
            if debug and debug_info:
                metadata['debug_info'] = debug_info
            
            # Additional validation
            if warped is not None:
                if not isinstance(warped, np.ndarray):
                    logger.warning(f"Expected numpy array for warped image, got {type(warped)}")
                    warped = None
                    metadata['success'] = False
                    metadata['error'] = 'Invalid warped image type'
                elif warped.size == 0:
                    logger.warning("Warped image is empty")
                    warped = None
                    metadata['success'] = False
                    metadata['error'] = 'Empty warped image'
            
            return warped, metadata
            
        except Exception as e:
            logger.error(f"Error extracting MRZ: {str(e)}")
            return None, {
                'success': False,
                'error': str(e),
                'ocr_available': TESSERACT_AVAILABLE,
                'mrz_detected': False
            }
        
    def validate_mrz_text(self, text: str) -> bool:
        """Validate if extracted text looks like MRZ format."""
        return self._is_mrz_like(text)
    
    def _is_mrz_like(self, text: str) -> bool:
        """Check if text matches MRZ pattern."""
        if not text:
            return False
        
        lines = [re.sub(r'[^A-Z0-9<]', '', ln.upper()) for ln in text.splitlines() if ln.strip()]
        if len(lines) < 2:
            return False
            
        candidate_lines = [ln for ln in lines if 10 <= len(ln) <= 44]
        if len(candidate_lines) < 2:
            return False
            
        checks = sum(1 for ln in candidate_lines[:2] if self.MRZ_REGEX.match(ln))
        return checks >= 1

    def _order_points_clockwise(self, pts: np.ndarray) -> np.ndarray:
        """Order points in clockwise order: top-left, top-right, bottom-right, bottom-left."""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left
        return rect

    def _four_point_warp(self, image: np.ndarray, pts: np.ndarray, dst_w: int = None, dst_h: int = None) -> np.ndarray:
        """Apply perspective transformation to get bird's eye view."""
        rect = self._order_points_clockwise(pts)
        (tl, tr, br, bl) = rect
        
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = int(max(widthA, widthB)) if dst_w is None else dst_w
        
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = int(max(heightA, heightB)) if dst_h is None else dst_h
        
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")
        
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped

    def _merge_contours_points(self, contours: list) -> np.ndarray:
        """Merge multiple contours into single contour format."""
        pts = np.vstack([c.reshape(-1, 2) for c in contours])
        return pts.reshape(-1, 1, 2).astype(np.int32)

    def _detect_and_crop_mrz(
        self,
        image: np.ndarray,
        debug: bool = False,
        use_ocr: bool = False
    ) -> Optional[np.ndarray]:
        """Core MRZ detection and cropping logic (fixed-percentage crop, no contours)."""
        orig = image.copy()
        orig_h, orig_w = orig.shape[:2]

        # Resize for processing (same as before)
        scale = 1.0
        if self.resize_width and orig_w != self.resize_width:
            scale = self.resize_width / float(orig_w)
            proc = cv2.resize(orig, (self.resize_width, int(orig_h * scale)), interpolation=cv2.INTER_AREA)
        else:
            proc = orig.copy()

        # Desaturate + sharpen (same as before)
        hsv = cv2.cvtColor(proc, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s[:] = 0
        hsv_desat = cv2.merge([h, s, v])
        proc = cv2.cvtColor(hsv_desat, cv2.COLOR_HSV2BGR)

        kernel_sharpening = np.array([[-1, -1, -1],
                                    [-1,  9, -1],
                                    [-1, -1, -1]])
        proc = cv2.filter2D(proc, -1, kernel_sharpening)

        # keep proc dims
        h, w = proc.shape[:2]

        # Image processing pipeline (same as before) - produce 'closed' for debug
        gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 7))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kernel)

        gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        if maxVal - minVal != 0:
            gradX = ((gradX - minVal) / (maxVal - minVal) * 255).astype("uint8")
        else:
            gradX = np.zeros_like(gradX, dtype="uint8")

        gradX = cv2.GaussianBlur(gradX, (3, 3), 0)
        _, thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        sq_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sq_kernel)
        closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

        # -------------------------
        # REPLACEMENT: fixed-percentage crop (no contours)
        # -------------------------
        # Percentages you asked for:
        x_min_proc = int(0.01 * w)
        x_max_proc = int(0.99 * w)
        y_min_proc = int(0.75 * h)
        y_max_proc = h

        # Make sure the proc coords are valid
        x_min_proc = max(0, min(x_min_proc, w - 1))
        x_max_proc = max(0, min(x_max_proc, w))
        y_min_proc = max(0, min(y_min_proc, h - 1))
        y_max_proc = max(0, min(y_max_proc, h))

        # Convert crop coords back to original image coordinates if we resized earlier
        if scale != 1.0:
            scale_inv = 1.0 / scale
            x_min = int(round(x_min_proc * scale_inv))
            x_max = int(round(x_max_proc * scale_inv))
            y_min = int(round(y_min_proc * scale_inv))
            y_max = int(round(y_max_proc * scale_inv))
        else:
            x_min, x_max, y_min, y_max = x_min_proc, x_max_proc, y_min_proc, y_max_proc

        # Clamp to original bounds
        x_min = max(0, min(x_min, orig_w - 1))
        x_max = max(0, min(x_max, orig_w))
        y_min = max(0, min(y_min, orig_h - 1))
        y_max = max(0, min(y_max, orig_h))

        # Build box points in original coordinates (so warp works)
        box_orig = np.array([
            [x_min, y_min],
            [x_max - 1, y_min],
            [x_max - 1, y_max - 1],
            [x_min, y_max - 1]
        ], dtype="float32")

        # Expand bounding box by expand_pixels (safely)
        """xs, ys = box_orig[:, 0], box_orig[:, 1]
        x_min = max(int(xs.min()) - self.expand_pixels, 0)
        y_min = max(int(ys.min()) - self.expand_pixels, 0)
        x_max = min(int(xs.max()) + self.expand_pixels, orig_w - 1)
        y_max = min(int(ys.max()) + self.expand_pixels, orig_h - 1)"""

        # Attempt perspective warp with the rectangular box
        try:
            # Use the four corners in original coordinates for warping
            """box_for_warp = np.array([
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max]
            ], dtype="float32")"""
            """warped = self._four_point_warp(gray, box_for_warp)"""
            warped = proc[y_min_proc:y_max_proc, x_min_proc:x_max_proc]

        except Exception as e:
            logger.warning(f"Perspective warp failed, using rect crop: {e}")
            #warped = gray[y_min_proc:y_max_proc, x_min_proc:x_max_proc]

        # Handle narrow crops similarly to previous logic:
        # compute a pseudo h_c based on proc crop height (scaled to original)
        """h_c_proc = (y_max_proc - y_min_proc)
        if h_c_proc <= 0:
            h_c_proc = 1
        if warped is not None and warped.shape[0] < int(1.3 * (h_c_proc * (1.0 / max(scale, 1e-6)))):
            extra = int(h_c_proc * (1.5 / max(scale, 1e-6)))
            y_min2 = max(y_min - extra, 0)
            y_max2 = min(y_max + extra, orig_h - 1)
            warped = gray[y_min2:y_max2, x_min:x_max]"""

        # OCR validation if requested
        ocr_result = None
        if use_ocr and TESSERACT_AVAILABLE and warped is not None:
            ocr_result = self._perform_ocr_validation(warped)

        # Prepare metadata (keep merged_contours_count but set to 1 because no merging)
        metadata = {
            'mrz_detected': warped is not None,
            'merged_contours_count': 1,
            'ocr_result': ocr_result,
            'image_dimensions': {'height': orig_h, 'width': orig_w}
        }

        if debug:
            debug_info = {
                "processed": proc, "gray": gray, "blackhat": blackhat,
                "gradX": gradX, "thresh": thresh, "closed": closed,
                "selected_candidate_box": (x_min_proc, y_min_proc, x_max_proc - x_min_proc, y_max_proc - y_min_proc),
                "merged_contours_count": 1,
                "expanded_bbox": (x_min, y_min, x_max - x_min, y_max - y_min),
            }
            metadata['debug_info'] = debug_info

        return warped, metadata


    def _perform_ocr_validation(self, image: np.ndarray) -> Dict[str, Any]:
        """Perform OCR on MRZ region and validate format."""
        try:
            ocr_config = "--oem 1 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
            ocr_text = pytesseract.image_to_string(image, config=ocr_config)
            is_valid = self._is_mrz_like(ocr_text)
            
            return {
                'text': ocr_text,
                'is_mrz_like': is_valid,
                'success': True
            }
        except Exception as e:
            logger.error(f"OCR validation failed: {e}")
            return {
                'text': None,
                'is_mrz_like': False,
                'success': False,
                'error': str(e)
            }
