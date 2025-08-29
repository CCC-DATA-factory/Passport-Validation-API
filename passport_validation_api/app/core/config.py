import os
from dotenv import load_dotenv
from typing import Dict, Any, Optional

load_dotenv()

class Settings:
    TESSERACT_CMD: str = os.getenv("TESSERACT_CMD", "")
    # Add other settings here, e.g. storage paths, thresholds
    mrz_config: Dict[str, Any] = {
        'resize_width': 1000,
        'expand_pixels': 10,
        'vertical_merge_gap_factor': 1.8
    }
    quality_config: Dict[str, Any] = {
        'min_confidence': 0.7,
        'max_noise': 0.2,
        'min_length': 10
    }
settings = Settings()