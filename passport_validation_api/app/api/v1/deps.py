import os
from dotenv import load_dotenv
from fastapi import Depends, HTTPException

# Load .env on import
load_dotenv()

TESSERACT_CMD = os.getenv("TESSERACT_CMD")
if not TESSERACT_CMD:
    raise RuntimeError("TESSERACT_CMD not defined in .env")

from pytesseract import pytesseract
pytesseract.tesseract_cmd = TESSERACT_CMD


def get_tesseract_cmd():
    """Dependency to inject the tesseract command path."""
    return TESSERACT_CMD