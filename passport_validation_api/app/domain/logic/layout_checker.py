import cv2
import numpy as np
from typing import Tuple

#-----------------------------------------------------V1---------------------------------------------------


def check_passport_layout(image: np.ndarray) -> Tuple[bool, str]:
    """
    Return (ok: bool, message: str)
    Synchronous function — do not make it async (cv2 is blocking).
    """
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return False, "No face detected in passport photo"

    x, _, w, _ = faces[0]
    # heuristic: face should be on right side (change if needed)
    if x > width * 0.4:
        return False, "Face position incorrect (should be on right side)"

    return True, "Basic layout OK"

#-----------------------------------------------------V2---------------------------------------------------


import cv2
import numpy as np
import re


class LayoutChecker:
    async def check_passport_layout(self, image: np.ndarray) -> dict:
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Load Haar Cascade
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if not len(faces):
            return {"is_valid_layout": False, "message": "No face detected in passport photo"}

        # Pick largest detected face (usually the passport face)
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_center_x = x + w / 2
        face_center_y = y + h / 2

        # --- Horizontal rule (face should be on LEFT side) ---
        min_x = width * 0.15   # 15% of width
        max_x = width * 0.45   # 45% of width

        if not (min_x <= face_center_x <= max_x):
            return {
                "is_valid_layout": False,
                "message": f"Face position incorrect (expected between {int(min_x)}px and {int(max_x)}px, got {int(face_center_x)}px)"
            }

        # --- Vertical rule (face should be upper half, not near MRZ) ---
        if face_center_y > height * 0.71:
            return {
                "is_valid_layout": False,
                "message": f"Face too low in image (y={int(face_center_y)}px, should be above {int(height*0.71)}px)"
            }

        # --- Size rule (face should be big enough, but not full frame) ---
        face_area = w * h
        img_area = width * height
        face_ratio = face_area / img_area

        if face_ratio < 0.01:  # too small
            return {
                "is_valid_layout": False,
                "message": f"Face too small (only {face_ratio*100:.1f}% of image area)"
            }
        if face_ratio > 0.4:  # too large
            return {
                "is_valid_layout": False,
                "message": f"Face too large (covers {face_ratio*100:.1f}% of image area)"
            }

        # ✅ If all checks passed
        return {"is_valid_layout": True, "message": "Layout OK (face on left, proper position & size)"}

