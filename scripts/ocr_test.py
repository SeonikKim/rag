import pytesseract
from PIL import Image
import os

# (선택) Tesseract 경로 지정 (환경변수 PATH 안 잡혔을 경우만 필요)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# # OCR 실행할 이미지 경로
# img_path = "./out/p0001.png"

# # 한국어+영어 OCR
# text = pytesseract.image_to_string(Image.open(img_path), lang="kor+eng")

# print("=== OCR 결과 ===")
# print(text)
