from pydantic import BaseModel
from typing import List, Optional


class DetectionImageResponse(BaseModel):
    imageId: str
    originalFilename: str
    mimeType: str
    sizeBytes: int
    width: int
    height: int
    originalPath: str
    processedPath: str
    status: str


class CreateDetectionSessionResponse(BaseModel):
    sessionId: str
    status: str
    images: List[DetectionImageResponse]


class ErrorResponse(BaseModel):
    detail: str