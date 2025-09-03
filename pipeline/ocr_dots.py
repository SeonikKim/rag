# -*- coding: utf-8 -*-
"""간단한 OCR 래퍼.

외부 dots.ocr 엔진 대신 Tesseract OCR(pytesseract)을 사용하여
이미지에서 텍스트와 바운딩 박스를 추출한다.
"""

from typing import Dict, List

try:  # optional imports; raise clearer error if missing
    from PIL import Image
    import pytesseract
except Exception as e:  # pragma: no cover - optional dependency
    Image = None  # type: ignore
    pytesseract = None  # type: ignore


class DotsOCR:
    def __init__(self, **kwargs):
        self.opts = kwargs
        if Image is None or pytesseract is None:
            raise ImportError("pytesseract and Pillow are required for DotsOCR")

        # 하드코딩된 Tesseract 실행 파일 경로
        pytesseract.pytesseract.tesseract_cmd = (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        )

    def run(self, image_path: str) -> Dict:
        """Run OCR on the given image and return block-level results."""

        img = Image.open(image_path)
        lang = self.opts.get("lang", "kor+eng")

        data = pytesseract.image_to_data(
            img, lang=lang, output_type=pytesseract.Output.DICT
        )

        blocks: List[Dict] = []
        n = len(data["text"])
        for i in range(n):
            text = data["text"][i].strip()
            if not text:
                continue
            x, y = data["left"][i], data["top"][i]
            w, h = data["width"][i], data["height"][i]
            conf_str = data["conf"][i]
            try:
                conf = float(conf_str) / 100.0 if conf_str != "-1" else 0.0
            except ValueError:
                conf = 0.0

            blocks.append(
                {
                    "id": f"b{i}",
                    "type": "paragraph",
                    "bbox": [x, y, x + w, y + h],
                    "text": text,
                    "conf": conf,
                }
            )

        avg_conf = (
            sum(b["conf"] for b in blocks) / len(blocks) if blocks else 0.0
        )

        return {"page": 1, "avg_conf": avg_conf, "blocks": blocks, "lang": lang}
