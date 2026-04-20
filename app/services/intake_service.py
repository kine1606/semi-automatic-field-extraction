import shutil

from fastapi import UploadFile

from app.services.image_validation import validate_files
from app.services.image_processing import (
    generate_session_id,
    generate_image_id,
    ensure_session_dirs,
    normalize_image,
)


async def create_detection_session(files: list[UploadFile]) -> dict:
    await validate_files(files)

    session_id = generate_session_id()
    original_dir, processed_dir = ensure_session_dirs(session_id)

    images = []

    for file in files:
        await file.seek(0)

        image_id = generate_image_id()
        ext = (file.filename.rsplit(".", 1)[1].lower() if file.filename and "." in file.filename else "jpg")

        original_path = original_dir / f"{image_id}.{ext}"
        processed_path = processed_dir / f"{image_id}.jpg"

        with open(original_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        width, height = normalize_image(original_path, processed_path)

        images.append({
            "imageId": image_id,
            "originalFilename": file.filename,
            "mimeType": file.content_type,
            "sizeBytes": original_path.stat().st_size,
            "width": width,
            "height": height,
            "originalPath": str(original_path),
            "processedPath": str(processed_path),
            "status": "READY_FOR_CLASSIFICATION",
        })

    return {
        "sessionId": session_id,
        "status": "READY",
        "images": images,
    }