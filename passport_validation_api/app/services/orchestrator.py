import numpy as np
import cv2
from typing import Dict, Any, Optional
from fastapi import UploadFile
from app.utils.image_io import decode_upload
from app.utils.mrz_utils import correct_mrz_name_filler, apply_name_corrections_to_parsed
from app.domain.logic.quality_checker import QualityChecker
from app.domain.logic.mrz_extractor import validate_mrz
from app.domain.logic.layout_checker import LayoutChecker
from app.domain.models.mrz_data import MRZData
from app.domain.logic.mrz_extractor import MRZExtractor
from app.domain.models.mrz_data import MRZDataV2
from io import BytesIO
import asyncio
from app.domain.logic.layout_checker import check_passport_layout
from app.domain.logic.quality_checker import check_image_quality
from app.core.config import Settings            
import logging
import re
import base64
import inspect


try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logging.warning("Tesseract not available. OCR validation will be disabled.")

logger = logging.getLogger(__name__)

#---------------------------------------------------------V1----------------------------------------------------------


async def process_passport_image(file: UploadFile) -> dict:
    contents = await file.read()
    image = decode_upload(contents)
    if image is None:
        raise ValueError("Invalid image file")

    # Quality
    ok, msg = check_image_quality(image)
    if not ok:
        raise ValueError(msg)

    # MRZ
    ok, mrz_map = validate_mrz(image)
    if not ok:
        raise ValueError("MRZ not found or invalid")
    mrz_obj = MRZData(**mrz_map)

    # Layout
    ok, layout_msg = check_passport_layout(image)
    if not ok:
        raise ValueError(layout_msg)

    # Success
    return {
        "valid": True,
        "data": mrz_obj.dict(),
        "checks": {
            "image_quality": msg,
            "layout_validation": layout_msg
        }
    }



#---------------------------------------------------------V2----------------------------------------------------------



class PassportValidationOrchestrator:
    """Orchestrates the entire passport validation process."""
    
    def __init__(self):
        self.settings = Settings()
        self.mrz_extractor = MRZExtractor(config=self.settings.mrz_config)
        self.quality_checker = QualityChecker()
        self.layout_checker = LayoutChecker()


    def _clean_numpy_from_dict(self, obj: Any) -> Any:
        """
        Recursively convert numpy types, ndarrays and bytes into JSON-serializable Python types.
        - np.ndarray: converted to Python lists OR base64-encoded PNG if it looks like an image (uint8 with 2-3 dims)
        - numpy scalars (np.generic): convert via .item()
        - bytes: try to decode utf-8, otherwise base64-encode
        - dict/list/tuple: cleaned recursively
        """
        # dict
        if isinstance(obj, dict):
            return {k: self._clean_numpy_from_dict(v) for k, v in obj.items()}

        # list / tuple
        if isinstance(obj, (list, tuple)):
            return [self._clean_numpy_from_dict(v) for v in obj]

        # numpy ndarray
        if isinstance(obj, np.ndarray):
            try:
                # If it looks like an image (uint8 and 2 or 3 dims), encode as PNG base64
                if obj.dtype == np.uint8 and obj.ndim in (2, 3):
                    try:
                        _, buf = cv2.imencode('.png', obj)
                        return base64.b64encode(buf).decode('utf-8')
                    except Exception:
                        return obj.tolist()
                # Otherwise convert to list
                return obj.tolist()
            except Exception:
                # Fallback
                return obj.tolist() if hasattr(obj, "tolist") else str(obj)

        # numpy scalar types (np.generic covers np.float64, np.int64, etc.)
        if isinstance(obj, np.generic):
            try:
                return obj.item()
            except Exception:
                # Last-resort fallback
                try:
                    # For boolean-like numpy scalars
                    if hasattr(obj, "__bool__"):
                        return bool(obj)
                except Exception:
                    pass
                return str(obj)

        # bytes
        if isinstance(obj, (bytes, bytearray)):
            try:
                return obj.decode('utf-8')
            except Exception:
                return base64.b64encode(bytes(obj)).decode('utf-8')

        # Everything else (str, int, float, None, bool)
        return obj
    
    # helper to normalize a potentially non-dict quality_result
    def _normalize_quality_result(self, raw: Any) -> dict:
        # if already dict, return as-is
        if isinstance(raw, dict):
            return raw
        # tuple or list like (ok, message)
        if isinstance(raw, (tuple, list)) and len(raw) >= 1:
            ok = bool(raw[0])
            msg = str(raw[1]) if len(raw) > 1 else ""
            return {
                "is_acceptable": ok,
                "is_valid": ok,
                "message": msg
            }
        # boolean
        if isinstance(raw, bool):
            return {
                "is_acceptable": raw,
                "is_valid": raw,
                "message": ""
            }
        # fallback
        return {
            "is_acceptable": False,
            "is_valid": False,
            "message": f"Unexpected quality result type: {type(raw).__name__}"
        }


    async def validate_passport(
        self, 
        image: np.ndarray, 
        include_debug: bool = False
    ) -> Dict[str, Any]:
        """
        Complete passport validation workflow.
        
        Args:
            image: Input passport image as numpy array
            include_debug: Whether to include debug information
            
        Returns:
            Dictionary containing validation results
        """
        validation_result = {
            'is_valid': False,
            'validation_steps': {},
            'errors': [],
            'mrz_data': None
        }
        
        try:
            # Step 1: Quality Check (robust to sync/async helpers)
            logger.info("Starting quality validation...")
            quality_raw = self.quality_checker.check_image_quality(image)
            if inspect.isawaitable(quality_raw):
                quality_raw = await quality_raw
            quality_result = self._normalize_quality_result(quality_raw)
            logger.debug("Normalized quality result: %s", quality_result)
            validation_result['validation_steps']['quality'] = quality_result

            if not quality_result.get('is_acceptable', quality_result.get('is_valid', False)):
                validation_result['errors'].append('Image quality insufficient for processing')
                return validation_result

            # Step 2: Layout Check (robust to sync/async)
            logger.info("Starting layout validation...")
            layout_raw = self.layout_checker.check_passport_layout(image)
            if inspect.isawaitable(layout_raw):
                layout_raw = await layout_raw

            # normalize layout_result (simple pattern)
            if isinstance(layout_raw, dict):
                layout_result = layout_raw
            else:
                # fallback: treat truthy value as valid layout
                layout_result = {
                    "is_valid_layout": bool(layout_raw),
                    "is_valid": bool(layout_raw),
                    "message": ""
                }
            logger.debug("Layout result: %s", layout_result)
            validation_result['validation_steps']['layout'] = layout_result

            if not layout_result.get('is_valid_layout', layout_result.get('is_valid', False)):
                validation_result['errors'].append('Passport layout validation failed')
                return validation_result

            # Step 3: MRZ extraction (robust to async)
            logger.info("Starting MRZ extraction...")
            mrz_raw = self.mrz_extractor.extract_mrz_region(
                image,
                use_ocr=True,
                debug=include_debug
            )
            if inspect.isawaitable(mrz_raw):
                mrz_raw = await mrz_raw

            # expect either (img, metadata) or dict containing both
            mrz_image, mrz_metadata = None, {}
            if isinstance(mrz_raw, tuple) or isinstance(mrz_raw, list):
                # unpack safely
                if len(mrz_raw) >= 1:
                    mrz_image = mrz_raw[0]
                if len(mrz_raw) >= 2:
                    mrz_metadata = mrz_raw[1] or {}
            elif isinstance(mrz_raw, dict):
                # dict may have keys 'mrz_image' and 'mrz_metadata'
                mrz_image = mrz_raw.get('mrz_image')
                mrz_metadata = mrz_raw.get('mrz_metadata', {})
            else:
                mrz_image = None
                mrz_metadata = {}

            # clean numpy & ensure JSON serializability (use your _clean_numpy_from_dict)
            clean_metadata = self._clean_numpy_from_dict(mrz_metadata)

            # debug image -> base64
            if include_debug and mrz_image is not None:
                _, buffer = cv2.imencode('.png', mrz_image)
                mrz_image_base64 = base64.b64encode(buffer).decode('utf-8')
                clean_metadata['mrz_image_base64'] = mrz_image_base64

            validation_result['validation_steps']['mrz_extraction'] = clean_metadata

            if mrz_image is None:
                validation_result['errors'].append('MRZ region not detected')
                return validation_result
            
            # Step 4: MRZ Data Parsing (you'll implement this)
            logger.info("Parsing MRZ data...")
            mrz_data = await self._parse_mrz_data_1(mrz_image)
            
            if mrz_data:
                validation_result['mrz_data'] = mrz_data.dict()
                validation_result['is_valid'] = True
                logger.info("Passport validation completed successfully")
            else:
                validation_result['errors'].append('MRZ data parsing failed')
            
        except Exception as e:
            logger.error(f"Validation failed with error: {str(e)}")
            validation_result['errors'].append(f'Validation error: {str(e)}')
        
        return validation_result
    
    async def _parse_mrz_data(self, mrz_image: np.ndarray) -> Optional[MRZDataV2]:

        """
        Parse MRZ data from extracted image using OCR and ICAO format validation.
        """
        try:
            if not TESSERACT_AVAILABLE:
                logger.error("Tesseract not available for MRZ parsing")
                return None
            
            # Step 1: OCR to extract text
            ocr_config = "--oem 1 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
            raw_text = pytesseract.image_to_string(mrz_image, config=ocr_config)
            
            # Clean and split into lines
            lines = [line.strip().upper() for line in raw_text.splitlines() if line.strip()]
            
            # Step 2: Validate we have exactly 2 MRZ lines
            if len(lines) < 2:
                logger.warning(f"Expected 2 MRZ lines, got {len(lines)}")
                return None
            
            # Take the first two valid-looking lines
            mrz_lines = []
            for line in lines:
                # Clean line: remove non-MRZ characters
                clean_line = re.sub(r'[^A-Z0-9<]', '', line)
                if len(clean_line) >= 44:  # Standard passport MRZ length
                    mrz_lines.append(clean_line[:44])  # Trim to exact length
                if len(mrz_lines) == 2:
                    break
            
            if len(mrz_lines) != 2:
                logger.warning("Could not find 2 valid MRZ lines")
                return None
                
            line1, line2 = mrz_lines
            
            # Step 3: Parse according to ICAO 9303 specifications
            parsed_data = self._parse_icao_mrz(line1, line2)
            
            if not parsed_data:
                logger.warning("MRZ parsing failed - invalid format")
                return None
            
            # Step 4: Validate check digits
            if not self._validate_check_digits(line1, line2, parsed_data):
                logger.warning("MRZ check digit validation failed")
                # Still return data but mark as potentially invalid
                parsed_data['check_digit_valid'] = False
            else:
                parsed_data['check_digit_valid'] = True
            
            # Create MRZData object
            return MRZDataV2(

                document_type=parsed_data['document_type'],
                country_code=parsed_data['country_code'],
                surname=parsed_data['surname'],
                given_names=parsed_data['given_names'],
                passport_number=parsed_data['passport_number'],
                nationality=parsed_data['nationality'],
                date_of_birth=parsed_data['date_of_birth'],
                gender=parsed_data['gender'],
                expiration_date=parsed_data['expiration_date'],
                personal_number=parsed_data['personal_number'],
                raw_mrz_line1=line1,
                raw_mrz_line2=line2,
                check_digit_valid=parsed_data['check_digit_valid']
            )
            
        except Exception as e:
            logger.error(f"MRZ parsing failed: {e}")
            return None
    
    def _parse_icao_mrz(self, line1: str, line2: str) -> Optional[Dict[str, str]]:
        """
        Parse MRZ lines according to ICAO 9303 standard.
        
        Line 1: P<ISSUER<SURNAME<<GIVEN_NAMES<<<<<<<<<<<<<<<
        Line 2: PASSPORT_NO<CHECK<NATIONALITY<DOB<GENDER<EXP<PERSONAL_NO<CHECK<OVERALL_CHECK
        """
        try:
            # Validate line lengths
            if len(line1) != 44 or len(line2) != 44:
                return None
            
            # Parse Line 1
            document_type = line1[0]  # Should be 'P' for passport
            country_code = line1[2:5].replace('<', '')
            
            # Extract name portion (positions 5-44)
            name_portion = line1[5:44]
            # Split on '<<' to separate surname from given names
            if '<<' in name_portion:
                name_parts = name_portion.split('<<')
                surname = name_parts[0].replace('<', ' ').strip()
                given_names = name_parts[1].replace('<', ' ').strip() if len(name_parts) > 1 else ''
            else:
                # Fallback: look for single '<' separators
                name_clean = name_portion.replace('<', ' ')
                name_words = [w for w in name_clean.split() if w]
                surname = name_words[0] if name_words else ''
                given_names = ' '.join(name_words[1:]) if len(name_words) > 1 else ''
            
            # Parse Line 2
            passport_number = line2[0:9].replace('<', '')
            passport_check = line2[9]
            nationality = line2[10:13].replace('<', '')
            date_of_birth = line2[13:19]
            dob_check = line2[19]
            gender = line2[20]
            expiration_date = line2[21:27]
            exp_check = line2[27]
            personal_number = line2[28:42].replace('<', '')
            personal_check = line2[42]
            overall_check = line2[43]
            
            return {
                'document_type': document_type,
                'country_code': country_code,
                'surname': surname,
                'given_names': given_names,
                'passport_number': passport_number,
                'nationality': nationality,
                'date_of_birth': date_of_birth,
                'gender': gender,
                'expiration_date': expiration_date,
                'personal_number': personal_number,
                'passport_check': passport_check,
                'dob_check': dob_check,
                'exp_check': exp_check,
                'personal_check': personal_check,
                'overall_check': overall_check
            }
            
        except Exception as e:
            logger.error(f"ICAO MRZ parsing error: {e}")
            return None
    
    def _calculate_check_digit(self, data: str) -> str:
        """
        Calculate MRZ check digit using ICAO algorithm.
        Weights: 7, 3, 1, 7, 3, 1, ... (repeating)
        """
        weights = [7, 3, 1]
        total = 0
        
        for i, char in enumerate(data):
            if char.isdigit():
                value = int(char)
            elif char.isalpha():
                value = ord(char) - ord('A') + 10
            elif char == '<':
                value = 0
            else:
                continue
                
            total += value * weights[i % 3]
        
        return str(total % 10)
    
    def _validate_check_digits(self, line1: str, line2: str, parsed_data: Dict[str, str]) -> bool:
        """
        Validate all MRZ check digits according to ICAO standards.
        """
        try:
            # Passport number check digit
            passport_check_calc = self._calculate_check_digit(parsed_data['passport_number'])
            if passport_check_calc != parsed_data['passport_check']:
                logger.warning(f"Passport number check digit mismatch: {passport_check_calc} vs {parsed_data['passport_check']}")
                return False
            
            # Date of birth check digit
            dob_check_calc = self._calculate_check_digit(parsed_data['date_of_birth'])
            if dob_check_calc != parsed_data['dob_check']:
                logger.warning(f"DOB check digit mismatch: {dob_check_calc} vs {parsed_data['dob_check']}")
                return False
            
            # Expiration date check digit
            exp_check_calc = self._calculate_check_digit(parsed_data['expiration_date'])
            if exp_check_calc != parsed_data['exp_check']:
                logger.warning(f"Expiration check digit mismatch: {exp_check_calc} vs {parsed_data['exp_check']}")
                return False
            
            # Personal number check digit (if personal number exists)
            if parsed_data['personal_number']:
                personal_check_calc = self._calculate_check_digit(parsed_data['personal_number'])
                if personal_check_calc != parsed_data['personal_check']:
                    logger.warning(f"Personal number check digit mismatch: {personal_check_calc} vs {parsed_data['personal_check']}")
                    return False
            
            # Overall check digit (most complex)
            # Includes: passport_number + passport_check + dob + dob_check + exp + exp_check + personal_number + personal_check
            overall_data = (
                parsed_data['passport_number'] + parsed_data['passport_check'] +
                parsed_data['date_of_birth'] + parsed_data['dob_check'] +
                parsed_data['expiration_date'] + parsed_data['exp_check'] +
                parsed_data['personal_number'] + parsed_data['personal_check']
            )
            
            overall_check_calc = self._calculate_check_digit(overall_data)
            if overall_check_calc != parsed_data['overall_check']:
                logger.warning(f"Overall check digit mismatch: {overall_check_calc} vs {parsed_data['overall_check']}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Check digit validation error: {e}")
            return False
    
    def _validate_date_format(self, date_str: str) -> bool:
        """Validate MRZ date format (YYMMDD)."""
        if len(date_str) != 6 or not date_str.isdigit():
            return False
        
        try:
            year = int(date_str[0:2])
            month = int(date_str[2:4])
            day = int(date_str[4:6])
            
            # Basic date validation
            if month < 1 or month > 12:
                return False
            if day < 1 or day > 31:
                return False
                
            return True
        except ValueError:
            return False


    async def _parse_mrz_data_1(self, mrz_image: np.ndarray) -> Optional[MRZDataV2]:
        """
        Parse MRZ data from extracted image using PassportEye (preferred) and fallback to Tesseract.
        Input: mrz_image as numpy.ndarray (BGR) or PIL image or bytes-like.
        Output: MRZDataV2 instance or None on failure.
        """
        try:
            # --- Input validation ---
            if mrz_image is None:
                logger.error("MRZ image is None")
                return None
                
            # --- Helpers ----------------------------------------------------------------
            def to_pil(img):
                """Convert different types to a PIL image safely (returns None on failure)."""
                try:
                    from PIL import Image as PILImage
                except Exception as e:
                    logger.debug("PIL not available: %s", e)
                    return None

                if img is None:
                    return None

                # numpy array (OpenCV BGR or grayscale)
                if isinstance(img, np.ndarray):
                    try:
                        if img.size == 0:  # Check for empty array
                            logger.debug("Empty numpy array")
                            return None
                            
                        if img.ndim == 3 and img.shape[2] == 3:
                            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        elif img.ndim == 3 and img.shape[2] == 4:
                            rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                        elif img.ndim == 2:  # grayscale
                            rgb = img
                        else:
                            logger.debug("Unsupported array shape: %s", img.shape)
                            return None
                        return PILImage.fromarray(rgb)
                    except Exception as e:
                        logger.debug("Failed converting numpy->PIL: %s", e)
                        return None

                # bytes (PNG/JPEG) or bytesIO
                if isinstance(img, (bytes, bytearray)):
                    try:
                        return PILImage.open(BytesIO(img))
                    except Exception as e:
                        logger.debug("Failed converting bytes->PIL: %s", e)
                        return None

                # If already a PIL image-ish (duck typed)
                if hasattr(img, "mode") and hasattr(img, "size") and hasattr(img, "tobytes"):
                    return img

                logger.debug("Unsupported image type for to_pil: %s", type(img))
                return None

            def clean_mrz_line(s: str) -> str:
                """Keep only MRZ charset and pad/trim to 44 chars."""
                if not s:
                    return ""
                s = re.sub(r'[^A-Z0-9<]', '', s.upper())
                if len(s) >= 44:
                    return s[:44]
                return s.ljust(44, '<')

            # Heuristic correction for digit-only fields (dates/checks)
            DIGIT_CORRECTIONS = str.maketrans({
                'O': '0', 'o': '0', 'Q': '0', 'D': '0',
                'I': '1', 'l': '1', 'L': '1',
                'Z': '2', 'z': '2',
                'S': '5', 's': '5',
                'G': '6', 'g': '6',
                'B': '8', 'b': '8'
            })

            def correct_digits_heuristic(s: str) -> str:
                if not s:
                    return s
                # Apply translation then keep only digits
                s = s.translate(DIGIT_CORRECTIONS)
                s = re.sub(r'[^0-9]', '', s)
                return s

            def normalize_yymmdd(s: str) -> str:
                """Return YYMMDD or empty string. Try to salvage common formats."""
                if not s:
                    return ""
                s = s.strip()
                digits = re.sub(r'[^0-9]', '', s)
                if len(digits) == 6:
                    # Validate the date before returning
                    try:
                        year = int(digits[0:2])
                        month = int(digits[2:4])
                        day = int(digits[4:6])
                        
                        # Basic validation
                        if month < 1 or month > 12:
                            return "000101"  # Default safe date
                        if day < 1 or day > 31:
                            return "000101"  # Default safe date
                        
                        return digits
                    except (ValueError, IndexError):
                        return "000101"
                        
                if len(digits) == 8:
                    return digits[2:]  # YYYYMMDD -> YYMMDD
                
                corrected = correct_digits_heuristic(s)
                if len(corrected) >= 6:
                    # Validate corrected date too
                    try:
                        year = int(corrected[0:2])
                        month = int(corrected[2:4])
                        day = int(corrected[4:6])
                        
                        if month < 1 or month > 12 or day < 1 or day > 31:
                            return "000101"
                        
                        return corrected[:6]
                    except (ValueError, IndexError):
                        return "000101"
                
                return "000101"  # Default safe date if nothing works

            # Convert input to PIL image first
            pil_img = to_pil(mrz_image)
            if pil_img is None:
                logger.error("Failed to convert input to PIL image")
                return None

            # Initialize variables
            parsed_data = {}
            raw_line1 = raw_line2 = None
            passporteye_obj = None

            # --- Try PassportEye first -----------------------------------------------
            passporteye_available = True
            try:
                from passporteye import read_mrz
            except ImportError as e:
                passporteye_available = False
                logger.debug("PassportEye import failed: %s", e)

            if passporteye_available:
                try:
                    passporteye_obj = read_mrz(pil_img)
                    logger.debug("PassportEye processing completed")
                except Exception as e:
                    logger.warning("PassportEye read_mrz raised: %s", e)
                    passporteye_obj = None

            # If PassportEye succeeded, try to extract lines/fields
            if passporteye_obj is not None:
                try:
                    # Get the raw MRZ lines first
                    mrz_data = passporteye_obj.mrz
                    if hasattr(mrz_data, 'aux') and hasattr(mrz_data.aux, 'image'):
                        # Get raw text lines
                        try:
                            raw_lines = []
                            if hasattr(passporteye_obj, 'mrz') and hasattr(passporteye_obj.mrz, 'raw_text'):
                                raw_text = passporteye_obj.mrz.raw_text
                                raw_lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
                            
                            if len(raw_lines) >= 2:
                                raw_line1, raw_line2 = raw_lines[0], raw_lines[1]
                            
                        except Exception as e:
                            logger.debug("Failed to get raw lines from PassportEye: %s", e)

                    # Extract parsed fields if available
                    try:
                        if hasattr(passporteye_obj, 'mrz') and passporteye_obj.mrz is not None:
                            mrz = passporteye_obj.mrz
                            
                            parsed_data = {
                                'document_type': getattr(mrz, 'document_type', '') or 'P',
                                'country_code': getattr(mrz, 'country', '') or '',
                                'surname': (getattr(mrz, 'surname', '') or '').replace('<', ' ').strip(),
                                'given_names': (getattr(mrz, 'names', '') or '').replace('<', ' ').strip(),
                                'passport_number': (getattr(mrz, 'number', '') or '').replace('<', ''),
                                'nationality': getattr(mrz, 'nationality', '') or '',
                                'date_of_birth': normalize_yymmdd(getattr(mrz, 'date_of_birth', '') or ''),
                                'gender': (getattr(mrz, 'sex', '') or '').upper(),
                                'expiration_date': normalize_yymmdd(getattr(mrz, 'expiration_date', '') or ''),
                                'personal_number': (getattr(mrz, 'personal_number', '') or '').replace('<', ''),
                            }
                            
                    except Exception as e:
                        logger.debug("Error extracting fields from PassportEye result: %s", e)

                except Exception as e:
                    logger.debug("Error processing PassportEye result: %s", e)

            # --- Fallback: If PassportEye failed to produce usable MRZ, run Tesseract OCR ---
            if not raw_line1 or not raw_line2:
                logger.debug("PassportEye did not yield usable MRZ lines — falling back to Tesseract OCR")
                try:
                    import pytesseract
                    # Fixed Tesseract configuration - use config parameter properly
                    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<'
                    raw_text = pytesseract.image_to_string(pil_img, config=custom_config)
                    lines = [l.strip().upper() for l in raw_text.splitlines() if l.strip()]
                    
                    # Find lines that look like MRZ (at least 40 chars, mostly alphanumeric)
                    mrz_lines = []
                    for line in lines:
                        cleaned = re.sub(r'[^A-Z0-9<]', '', line.upper())
                        if len(cleaned) >= 30:  # Relaxed minimum length
                            mrz_lines.append(cleaned)
                        if len(mrz_lines) == 2:
                            break
                    
                    if len(mrz_lines) >= 2:
                        raw_line1, raw_line2 = mrz_lines[0], mrz_lines[1]
                        logger.debug("Tesseract extracted MRZ lines successfully")
                    else:
                        logger.warning("Tesseract fallback did not yield usable MRZ lines")
                        
                except Exception as e:
                    logger.warning("Tesseract fallback failed: %s", e)

            # At this point, ensure we have raw lines
            if not raw_line1 or not raw_line2:
                logger.warning("Could not obtain two MRZ lines from PassportEye or Tesseract")
                # Return None instead of trying to create invalid object
                return None

            # Normalize/clean lines to MRZ charset and exact length
            line1 = clean_mrz_line(raw_line1)
            line2 = clean_mrz_line(raw_line2)

            # --- Extract from MRZ line positions if fields are missing ---
            try:
                # MRZ line 1: P<COUNTRY<<SURNAME<<GIVEN_NAMES<<<<<<<<<<<<<<<
                # MRZ line 2: PASSPORT_NUM_CHECK_NATIONALITY_DOB_CHECK_GENDER_EXPIRY_CHECK_PERSONAL_CHECK_OVERALL
                
                # Extract from line 1 if needed
                if line1.startswith('P<'):
                    country_from_line1 = line1[2:5].replace('<', '')
                    if not parsed_data.get('country_code'):
                        parsed_data['country_code'] = country_from_line1
                    
                    # Extract names from line 1
                    name_part = line1[5:].rstrip('<')
                    if '<<' in name_part:
                        parts = name_part.split('<<')
                        if not parsed_data.get('surname'):
                            parsed_data['surname'] = parts[0].replace('<', ' ').strip()
                        if len(parts) > 1 and not parsed_data.get('given_names'):
                            parsed_data['given_names'] = parts[1].replace('<', ' ').strip()

                # Extract from line 2 positions
                if len(line2) >= 44:
                    if not parsed_data.get('passport_number'):
                        parsed_data['passport_number'] = line2[0:9].replace('<', '')
                    if not parsed_data.get('passport_check'):
                        parsed_data['passport_check'] = line2[9] if len(line2) > 9 else '0'
                    if not parsed_data.get('nationality'):
                        parsed_data['nationality'] = line2[10:13].replace('<', '')
                    
                    # For your MRZ: Y847835<<7TUN9O903218F230907309892239<0204<86
                    # Position analysis:
                    # 0-8: Y847835<< (passport number)
                    # 9: 7 (passport check) 
                    # 10-12: TUN (nationality)
                    # 13-18: 9O9032 -> should be 900302 (DOB: March 2, 1990)
                    # 19: 1 (DOB check)
                    # 20: 8 -> should be F (gender, OCR error)
                    # 21-26: F23090 -> should be 230907 (Expiry: Sept 7, 2023)
                    # 27: 7 (expiry check)
                    # 28-41: 309892239<0204 (personal number)
                    
                    # Extract and correct DOB
                    raw_dob = line2[13:19] if len(line2) > 18 else ''
                    if raw_dob:
                        # Apply OCR corrections: O->0, common mistakes
                        corrected_dob = raw_dob.replace('O', '0').replace('o', '0')
                        corrected_dob = normalize_yymmdd(corrected_dob)
                        if not parsed_data.get('date_of_birth'):
                            parsed_data['date_of_birth'] = corrected_dob
                    
                    if not parsed_data.get('dob_check'):
                        parsed_data['dob_check'] = line2[19] if len(line2) > 19 else '0'
                    
                    # Extract and correct gender
                    if not parsed_data.get('gender'):
                        gender_char = line2[20] if len(line2) > 20 else 'X'
                        # Common OCR mistakes: 8->F, 6->F, etc.
                        if gender_char in ['8', '6', 'B']:
                            gender_char = 'F'
                        elif gender_char in ['1', 'I', 'l']:
                            gender_char = 'M'
                        parsed_data['gender'] = gender_char if gender_char in 'MFX' else 'X'
                    
                    # Extract and correct expiration date
                    if not parsed_data.get('expiration_date'):
                        # Looking at your line: F230907 - this seems to be at position 20-26
                        # But F is gender, so expiry should be at 21-26: 230907
                        raw_exp = line2[21:27] if len(line2) > 26 else ''
                        if raw_exp:
                            corrected_exp = raw_exp.replace('O', '0').replace('o', '0')
                            corrected_exp = normalize_yymmdd(corrected_exp)
                            parsed_data['expiration_date'] = corrected_exp
                    
                    if not parsed_data.get('exp_check'):
                        parsed_data['exp_check'] = line2[27] if len(line2) > 27 else '0'
                    
                    # Extract personal number (positions 28-41)
                    if not parsed_data.get('personal_number'):
                        personal_num = line2[28:42].replace('<', '') if len(line2) > 41 else ''
                        parsed_data['personal_number'] = personal_num
                    
                    if not parsed_data.get('personal_check'):
                        parsed_data['personal_check'] = line2[42] if len(line2) > 42 else '0'
                    if not parsed_data.get('overall_check'):
                        parsed_data['overall_check'] = line2[43] if len(line2) > 43 else '0'

            except Exception as e:
                logger.debug("Error extracting fields from MRZ fixed positions: %s", e)

            # --- Validation and defaults for required fields ---
            # Set defaults for required fields if they're empty
            parsed_data['document_type'] = parsed_data.get('document_type') or 'P'
            parsed_data['country_code'] = parsed_data.get('country_code') or 'XXX'
            parsed_data['surname'] = parsed_data.get('surname') or 'UNKNOWN'
            parsed_data['given_names'] = parsed_data.get('given_names') or 'UNKNOWN'
            parsed_data['passport_number'] = parsed_data.get('passport_number') or '000000000'
            parsed_data['nationality'] = parsed_data.get('nationality') or 'XXX'
            
            # Ensure dates are valid
            dob = parsed_data.get('date_of_birth') or '000101'
            if not dob or len(dob) != 6 or not dob.isdigit():
                dob = '000101'
            else:
                # Double-check date validity
                try:
                    year = int(dob[0:2])
                    month = int(dob[2:4])
                    day = int(dob[4:6])
                    if month < 1 or month > 12 or day < 1 or day > 31:
                        dob = '000101'
                except (ValueError, IndexError):
                    dob = '000101'
            parsed_data['date_of_birth'] = dob
            
            exp_date = parsed_data.get('expiration_date') or '991231'
            if not exp_date or len(exp_date) != 6 or not exp_date.isdigit():
                exp_date = '991231'
            else:
                # Double-check date validity
                try:
                    year = int(exp_date[0:2])
                    month = int(exp_date[2:4])
                    day = int(exp_date[4:6])
                    if month < 1 or month > 12 or day < 1 or day > 31:
                        exp_date = '991231'
                except (ValueError, IndexError):
                    exp_date = '991231'
            parsed_data['expiration_date'] = exp_date
            
            # Ensure gender is valid
            gender = (parsed_data.get('gender') or 'X').upper()
            if gender not in ['M', 'F', 'X']:
                gender = 'X'
            parsed_data['gender'] = gender
            
            parsed_data['personal_number'] = parsed_data.get('personal_number', '')

            # --- Handle check digits with proper corrections ---
            # Apply digit corrections to check digit fields
            check_fields = ['passport_check', 'dob_check', 'exp_check', 'personal_check', 'overall_check']
            for field in check_fields:
                if field not in parsed_data or not parsed_data[field]:
                    parsed_data[field] = '0'  # Default check digit
                else:
                    # Apply corrections and ensure single digit
                    corrected = correct_digits_heuristic(str(parsed_data[field]))
                    parsed_data[field] = corrected[:1] if corrected else '0'

            # Validate check digits
            try:
                checks_ok = self._validate_check_digits(line1, line2, parsed_data)
            except Exception as e:
                logger.debug("Exception during check-digit validation: %s", e)
                checks_ok = False

            # Build MRZDataV2 with validated data
            try:
                mrz_obj = MRZDataV2(
                    document_type=parsed_data['document_type'],
                    country_code=parsed_data['country_code'],
                    surname=parsed_data['surname'],
                    given_names=parsed_data['given_names'],
                    passport_number=parsed_data['passport_number'],
                    nationality=parsed_data['nationality'],
                    date_of_birth=parsed_data['date_of_birth'],
                    gender=parsed_data['gender'],
                    expiration_date=parsed_data['expiration_date'],
                    personal_number=parsed_data['personal_number'],
                    raw_mrz_line1=line1,
                    raw_mrz_line2=line2,
                    check_digit_valid=bool(checks_ok)
                )
                
                logger.info("Successfully created MRZDataV2 object")
                return mrz_obj

            except Exception as e:
                logger.error(f"Failed to create MRZDataV2 object: {e}")
                return None
            
        except Exception as e:
            logger.error(f"MRZ parsing (PassportEye + fallback) failed: {e}", exc_info=True)
            return None
        

