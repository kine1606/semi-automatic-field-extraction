from fastapi import UploadFile
from typing import List

from app.services.image_validation import validate_files
from app.services.image_processing import (
    generate_session_id,
    generate_image_id,
    ensure_session_dirs,
    save_upload_file,
)


async def create_detection_session(files: List[UploadFile]) -> dict:
    await validate_files(files)

    session_id = generate_session_id()
    original_dir, processed_dir = ensure_session_dirs(session_id)

    images = []

    for file in files:
        image_id = generate_image_id()
        image_data = await save_upload_file(
            upload_file=file,
            original_dir=original_dir,
            processed_dir=processed_dir,
            image_id=image_id,
        )
        images.append(image_data)

    return {
        "sessionId": session_id,
        "status": "READY",
        "images": images,
    }