from pydantic import BaseModel, Field, validator
from typing import Dict, Optional
from datetime import datetime, date
import re

#-----------------------------------------------------V1---------------------------------------------------


class MRZData(BaseModel):
    country: str
    passport_number: str
    birth_date: date
    expiry_date: date
    name: str
    surname: str
    gender: str
    nationality: str
    validation_details: dict = Field(default_factory=dict)


#----------------------------------------------V2---------------------------------------------------

class MRZDataV2(BaseModel):

    """Pydantic model for parsed MRZ data from passport."""
    
    document_type: str = Field(..., description="Document type (usually 'P' for passport)")
    country_code: str = Field(..., description="Issuing country code (3 letters)")
    surname: str = Field(..., description="Primary identifier (surname)")
    given_names: str = Field(..., description="Secondary identifier (given names)")
    passport_number: str = Field(..., description="Passport number")
    nationality: str = Field(..., description="Nationality code (3 letters)")
    date_of_birth: str = Field(..., description="Date of birth (YYMMDD)")
    gender: str = Field(..., description="Gender (M/F/X)")
    expiration_date: str = Field(..., description="Passport expiration date (YYMMDD)")
    personal_number: Optional[str] = Field("", description="Personal number (optional)")
    raw_mrz_line1: str = Field(..., description="Raw first MRZ line")
    raw_mrz_line2: str = Field(..., description="Raw second MRZ line")
    check_digit_valid: bool = Field(True, description="Whether check digits are valid")
    
    @validator('document_type')
    def validate_document_type(cls, v):
        if v not in ['P', 'V']:  # P for passport, V for visa
            raise ValueError('Document type must be P or V')
        return v
    
    @validator('country_code', 'nationality')
    def validate_country_codes(cls, v):
        if len(v) != 3 or not v.isalpha():
            raise ValueError('Country codes must be 3 letters')
        return v.upper()
    
    @validator('date_of_birth', 'expiration_date')
    def validate_dates(cls, v):
        if len(v) != 6 or not v.isdigit():
            raise ValueError('Dates must be in YYMMDD format')
        
        # Basic date validation
        year = int(v[0:2])
        month = int(v[2:4])
        day = int(v[4:6])
        
        if month < 1 or month > 12:
            raise ValueError('Invalid month in date')
        if day < 1 or day > 31:
            raise ValueError('Invalid day in date')
            
        return v
    
    @validator('gender')
    def validate_gender(cls, v):
        if v not in ['M', 'F', 'X']:
            raise ValueError('Gender must be M, F, or X')
        return v
    
    @validator('raw_mrz_line1', 'raw_mrz_line2')
    def validate_mrz_lines(cls, v):
        if len(v) != 44:
            raise ValueError('MRZ lines must be exactly 44 characters')
        if not re.match(r'^[A-Z0-9<]{44}$', v):
            raise ValueError('MRZ lines can only contain A-Z, 0-9, and <')
        return v
    
    def to_readable_format(self) -> Dict[str, str]:
        """Convert MRZ data to human-readable format."""
        return {
            'Document Type': 'Passport' if self.document_type == 'P' else 'Visa',
            'Issuing Country': self.country_code,
            'Surname': self.surname,
            'Given Names': self.given_names,
            'Passport Number': self.passport_number,
            'Nationality': self.nationality,
            'Date of Birth': self._format_date(self.date_of_birth),
            'Gender': {'M': 'Male', 'F': 'Female', 'X': 'Unspecified'}[self.gender],
            'Expiration Date': self._format_date(self.expiration_date),
            'Personal Number': self.personal_number or 'Not specified',
            'Check Digits Valid': 'Yes' if self.check_digit_valid else 'No'
        }
    
    def _format_date(self, yymmdd: str) -> str:
        """Convert YYMMDD to readable date format."""
        try:
            year = int(yymmdd[0:2])
            # Assume years 00-30 are 2000s, 31-99 are 1900s (common convention)
            full_year = 2000 + year if year <= 30 else 1900 + year
            month = int(yymmdd[2:4])
            day = int(yymmdd[4:6])
            
            date_obj = datetime(full_year, month, day)
            return date_obj.strftime('%d/%m/%Y')
        except ValueError:
            return yymmdd  # Return original if parsing fails
    
    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "document_type": "P",
                "country_code": "USA",
                "surname": "SMITH",
                "given_names": "JOHN MICHAEL",
                "passport_number": "123456789",
                "nationality": "USA",
                "date_of_birth": "850315",
                "gender": "M",
                "expiration_date": "300315",
                "personal_number": "",
                "raw_mrz_line1": "P<USASMITH<<JOHN<MICHAEL<<<<<<<<<<<<<<<<<<<",
                "raw_mrz_line2": "1234567890USA8503159M3003159<<<<<<<<<<<<<<04",
                "check_digit_valid": True
            }
        }