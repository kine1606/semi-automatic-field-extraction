from fastapi import HTTPException, UploadFile
from typing import List

MAX_FILES = 3
MAX_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


async def validate_files(files: List[UploadFile]) -> None:
    if not files:
        raise HTTPException(status_code=400, detail="At least one image is required.")

    if len(files) > MAX_FILES:
        raise HTTPException(status_code=400, detail="Maximum 3 images are allowed.")

    for file in files:
        if file.content_type not in ALLOWED_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.content_type}"
            )

        contents = await file.read()
        size = len(contents)

        if size == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        if size > MAX_SIZE_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"File {file.filename} exceeds 10MB."
            )

        await file.seek(0)