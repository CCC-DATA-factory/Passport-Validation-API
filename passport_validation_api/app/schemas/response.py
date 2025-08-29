from pydantic import BaseModel, Field
from typing import Dict, Any

class PassportValidationResponse(BaseModel):
    valid: bool
    data: Dict[str, Any] = Field(default_factory=dict)
    checks: Dict[str, str]