import cv2
import numpy as np
from typing import Tuple


#-----------------------------------------------------V1---------------------------------------------------


def check_image_quality(image: np.ndarray) -> Tuple[bool, str]:
    """
    Return (ok: bool, message: str)
    """
    height, width = image.shape[:2]
    # Minimum resolution check
    if width < 400 or height < 200:
        return False, "Image resolution too low (min 600x400 required)"

    # Sharpness check
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    if sharpness < 50:
        return False, f"Image too blurry (score: {sharpness:.1f})"

    return True, "Image quality OK"

#-----------------------------------------------------V2---------------------------------------------------



class QualityChecker:
    def check_image_quality(self, image: np.ndarray) -> dict:
        height, width = image.shape[:2]

        # Minimum resolution check
        if width < 400 or height < 200:
            return {
                "is_acceptable": False,
                "is_valid": False,
                "message": "Image resolution too low (min 600x400 required)",
                "resolution": (width, height)
            }

        # Sharpness check
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sharpness = float(np.var(cv2.Laplacian(gray, cv2.CV_64F)))
        if sharpness < 50:
            return {
                "is_acceptable": False,
                "is_valid": False,
                "message": f"Image too blurry (score: {sharpness:.1f})",
                "resolution": (width, height),
                "sharpness": sharpness
            }

        return {
            "is_acceptable": True,
            "is_valid": True,
            "message": "Image quality OK",
            "resolution": (width, height),
            "sharpness": sharpness
        }

