from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

img_path = r"C:\Users\ADMIN\Desktop\nhan-nang-luong-chi-so-hieu-suat-cong-suat-tren-t (6)-800x576.jpg"

print("langs:")
print(pytesseract.get_languages(config=""))

img = Image.open(img_path)
text = pytesseract.image_to_string(img, lang="eng")
print("OCR TEXT:")
print(repr(text))