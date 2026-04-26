from app.services.ocr_service import extract_text_fast, extract_text_variants
from app.services.extraction.common import (
    find_known_brand,
    infer_category,
    make_candidate,
    merge_ocr_variants,
)
from app.services.extraction.energy_label_parser import parse_energy_label
from app.services.extraction.nameplate_parser import parse_nameplate


def _parse_raw_photo(raw_text: str) -> dict:
    fields = {}

    brand, brand_evidence = find_known_brand(raw_text)
    if brand:
        fields["brandName"] = make_candidate(brand, 0.58, "raw_photo_ocr", brand_evidence)

    category, category_evidence = infer_category(raw_text)
    if category:
        fields["categoryType"] = make_candidate(category, 0.62, "raw_photo_ocr", category_evidence)

    return fields


def extract_fields_for_image(
    image_type: str,
    image_path: str,
    ocr_text_from_classify: str | None = None,
) -> dict:
    if image_type == "energy_label_vn":
        raw_text = merge_ocr_variants(extract_text_variants(image_path))
        fields = parse_energy_label(raw_text)
        return {"imageType": image_type, "rawText": raw_text, "fields": fields}

    if image_type == "nameplate_label":
        raw_text = merge_ocr_variants(extract_text_variants(image_path))
        fields = parse_nameplate(raw_text)
        return {"imageType": image_type, "rawText": raw_text, "fields": fields}

    if image_type == "raw_equipment_photo":
        raw_text = ocr_text_from_classify or extract_text_fast(image_path)
        fields = _parse_raw_photo(raw_text)
        return {"imageType": image_type, "rawText": raw_text, "fields": fields}

    if image_type == "barcode_qr":
        return {"imageType": image_type, "rawText": "", "fields": {}}

    return {"imageType": image_type, "rawText": "", "fields": {}}