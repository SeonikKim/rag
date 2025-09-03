import pytesseract
from PIL import Image
import os

# Tesseract 경로 (필요하면 설정)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

img_dir = "./out"   # OCR 돌릴 이미지가 있는 폴더
files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".png",".jpg",".jpeg"))])

for fname in files:
    path = os.path.join(img_dir, fname)
    text = pytesseract.image_to_string(Image.open(path), lang="kor+eng")
    print(f"\n=== OCR 결과: {fname} ===")
    print(text[:500])  # 처음 500자만 출력 (너무 길면 잘라서)
    print("="*80)
