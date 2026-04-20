from pathlib import Path
from uuid import uuid4
from PIL import Image, ImageOps
from fastapi import UploadFile, HTTPException
import hashlib
import shutil


STORAGE_ROOT = Path("storage")


def generate_session_id() -> str:
    return f"det_{uuid4().hex}"


def generate_image_id() -> str:
    return f"img_{uuid4().hex}"


def ensure_session_dirs(session_id: str) -> tuple[Path, Path]:
    original_dir = STORAGE_ROOT / session_id / "original"
    processed_dir = STORAGE_ROOT / session_id / "processed"

    original_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    return original_dir, processed_dir


def get_extension(filename: str | None, content_type: str) -> str:
    if filename and "." in filename:
        return filename.rsplit(".", 1)[1].lower()

    if content_type == "image/png":
        return "png"
    if content_type == "image/webp":
        return "webp"
    return "jpg"


def compute_sha256(file_path: Path) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def normalize_image(input_path: Path, output_path: Path, max_side: int = 1600) -> tuple[int, int]:
    try:
        with Image.open(input_path) as img:
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")

            width, height = img.size

            scale = min(max_side / width, max_side / height, 1.0)
            new_width = int(width * scale)
            new_height = int(height * scale)

            if scale < 1.0:
                img = img.resize((new_width, new_height))

            img.save(output_path, format="JPEG", quality=90)

            return width, height

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")


async def save_upload_file(
    upload_file: UploadFile,
    original_dir: Path,
    processed_dir: Path,
    image_id: str,
) -> dict:
    extension = get_extension(upload_file.filename, upload_file.content_type or "")
    original_path = original_dir / f"{image_id}.{extension}"
    processed_path = processed_dir / f"{image_id}.jpg"

    try:
        with open(original_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    original_width, original_height = normalize_image(original_path, processed_path)
    sha256 = compute_sha256(original_path)
    size_bytes = original_path.stat().st_size

    return {
        "imageId": image_id,
        "originalFilename": upload_file.filename or f"{image_id}.{extension}",
        "mimeType": upload_file.content_type or "application/octet-stream",
        "sizeBytes": size_bytes,
        "width": original_width,
        "height": original_height,
        "originalPath": str(original_path),
        "processedPath": str(processed_path),
        "sha256": sha256,
        "status": "READY_FOR_CLASSIFICATION",
    }