from fastapi import HTTPException, UploadFile

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_SIZE_BYTES = 10 * 1024 * 1024


async def validate_files(files: list[UploadFile]) -> None:
    if not files:
        raise HTTPException(status_code=400, detail="At least one image is required.")

    if len(files) > 3:
        raise HTTPException(status_code=400, detail="Maximum 3 images are allowed.")

    for file in files:
        if file.content_type not in ALLOWED_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.content_type}"
            )

        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail=f"File {file.filename} is empty.")

        if len(content) > MAX_SIZE_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"File {file.filename} exceeds 10MB."
            )

        await file.seek(0)