import numpy as np
import cv2
from typing import Tuple
import numpy as np
import base64
from fastapi import UploadFile
from typing import Optional, Union
import logging
from PIL import Image
import io

logger = logging.getLogger(__name__)

#-----------------------------------------------------V1---------------------------------------------------


def decode_upload(contents: bytes):
    nparr = np.frombuffer(contents, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def bytes_to_bgr(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes")
    return img

def bgr_to_jpeg_bytes(img: np.ndarray, quality: int = 90) -> bytes:
    success, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        raise ValueError('Failed to encode image')
    return enc.tobytes()


def jpeg_bytes_to_base64_str(jpeg_bytes: bytes) -> str:
    return base64.b64encode(jpeg_bytes).decode('ascii')

#--------------------------------------------------------------V2---------------------------------------------------

SUPPORTED_FORMATS = {
    'image/jpeg', 'image/jpg', 'image/png', 'image/bmp', 
    'image/tiff', 'image/tif', 'image/webp'
}

async def load_image_from_upload(file: UploadFile) -> Optional[np.ndarray]:
    """
    Load and convert uploaded file to OpenCV BGR image array.
    
    Args:
        file: FastAPI UploadFile object
        
    Returns:
        numpy.ndarray: BGR image array for OpenCV, or None if loading failed
    """
    try:
        # Validate content type
        if file.content_type not in SUPPORTED_FORMATS:
            logger.error(f"Unsupported image format: {file.content_type}")
            return None
        
        # Read file content
        contents = await file.read()
        if not contents:
            logger.error("Empty file received")
            return None
        
        # Convert to numpy array
        image_array = np.frombuffer(contents, np.uint8)
        
        # Decode image using OpenCV
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            logger.error("Failed to decode image with OpenCV, trying PIL fallback")
            # Fallback to PIL if OpenCV fails
            image = load_image_with_pil_fallback(contents)
        
        # Reset file pointer for potential future reads
        await file.seek(0)
        
        if image is not None:
            logger.info(f"Successfully loaded image: {image.shape}")
        else:
            logger.error("Failed to load image with both OpenCV and PIL")
            
        return image
        
    except Exception as e:
        logger.error(f"Error loading image from upload: {str(e)}")
        return None

def load_image_with_pil_fallback(image_bytes: bytes) -> Optional[np.ndarray]:
    """
    Fallback method to load image using PIL and convert to OpenCV format.
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        numpy.ndarray: BGR image array, or None if failed
    """
    try:
        # Use PIL to open image
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert PIL to numpy array
        image_array = np.array(pil_image)
        
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        return image_bgr
        
    except Exception as e:
        logger.error(f"PIL fallback failed: {str(e)}")
        return None

def load_image_from_path(image_path: str) -> Optional[np.ndarray]:
    """
    Load image from file path.
    
    Args:
        image_path: Path to image file
        
    Returns:
        numpy.ndarray: BGR image array, or None if failed
    """
    try:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            logger.error(f"Failed to load image from path: {image_path}")
            return None
        
        logger.info(f"Successfully loaded image from {image_path}: {image.shape}")
        return image
        
    except Exception as e:
        logger.error(f"Error loading image from path {image_path}: {str(e)}")
        return None

def save_image(image: np.ndarray, output_path: str) -> bool:
    """
    Save OpenCV image to file.
    
    Args:
        image: BGR image array
        output_path: Path where to save the image
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        success = cv2.imwrite(output_path, image)
        if success:
            logger.info(f"Image saved successfully to {output_path}")
        else:
            logger.error(f"Failed to save image to {output_path}")
        return success
        
    except Exception as e:
        logger.error(f"Error saving image to {output_path}: {str(e)}")
        return False

def validate_image_dimensions(image: np.ndarray, min_width: int = 100, min_height: int = 100) -> bool:
    """
    Validate image has minimum dimensions.
    
    Args:
        image: Input image array
        min_width: Minimum required width
        min_height: Minimum required height
        
    Returns:
        bool: True if dimensions are acceptable
    """
    if image is None:
        return False
    
    height, width = image.shape[:2]
    
    if width < min_width or height < min_height:
        logger.warning(f"Image too small: {width}x{height}, minimum: {min_width}x{min_height}")
        return False
    
    return True

def resize_image_if_large(
    image: np.ndarray, 
    max_width: int = 2000, 
    max_height: int = 2000
) -> np.ndarray:
    """
    Resize image if it's larger than specified dimensions while maintaining aspect ratio.
    
    Args:
        image: Input image array
        max_width: Maximum allowed width
        max_height: Maximum allowed height
        
    Returns:
        numpy.ndarray: Resized image (or original if already small enough)
    """
    try:
        height, width = image.shape[:2]
        
        if width <= max_width and height <= max_height:
            return image
        
        # Calculate scaling factor
        scale_w = max_width / width
        scale_h = max_height / height
        scale = min(scale_w, scale_h)
        
        # Calculate new dimensions
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")
        return resized
        
    except Exception as e:
        logger.error(f"Error resizing image: {str(e)}")
        return image  # Return original on error

def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert BGR image to grayscale.
    
    Args:
        image: BGR image array
        
    Returns:
        numpy.ndarray: Grayscale image
    """
    try:
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            return image  # Already grayscale
    except Exception as e:
        logger.error(f"Error converting to grayscale: {str(e)}")
        return image

def enhance_image_contrast(image: np.ndarray) -> np.ndarray:
    """
    Enhance image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Args:
        image: Input image (grayscale or BGR)
        
    Returns:
        numpy.ndarray: Enhanced image
    """
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Convert back to BGR if original was BGR
        if len(image.shape) == 3:
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        return enhanced
        
    except Exception as e:
        logger.error(f"Error enhancing contrast: {str(e)}")
        return image

# Utility function for getting file size
async def get_file_size(file: UploadFile) -> int:
    """
    Get the size of uploaded file in bytes.
    
    Args:
        file: FastAPI UploadFile object
        
    Returns:
        int: File size in bytes
    """
    try:
        # Save current position
        current_pos = file.file.tell()
        
        # Seek to end and get position (file size)
        file.file.seek(0, 2)
        size = file.file.tell()
        
        # Reset to original position
        file.file.seek(current_pos)
        
        return size
        
    except Exception as e:
        logger.error(f"Error getting file size: {str(e)}")
        return 0

def validate_file_size(file_size: int, max_size_mb: int = 10) -> bool:
    """
    Validate if file size is within acceptable limits.
    
    Args:
        file_size: File size in bytes
        max_size_mb: Maximum allowed size in megabytes
        
    Returns:
        bool: True if size is acceptable
    """
    max_size_bytes = max_size_mb * 1024 * 1024
    
    if file_size > max_size_bytes:
        logger.warning(f"File too large: {file_size / (1024*1024):.2f}MB, max: {max_size_mb}MB")
        return False
    
    return True