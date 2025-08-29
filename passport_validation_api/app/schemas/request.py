from pydantic import BaseModel
from fastapi import UploadFile, File

# If you need complex validation, otherwise FastAPI handles UploadFile
class PassportUpload(BaseModel):
    file: UploadFile