from PIL import Image


def estimate_green_ratio(image_path: str) -> float:
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((200, 200))

        total = 0
        green_like = 0

        for r, g, b in img.getdata():
            total += 1
            if g > r + 20 and g > b + 20 and g > 70:
                green_like += 1

        return green_like / total if total else 0.0
    except Exception:
        return 0.0