# -*- coding: utf-8 -*-
"""간단한 OCR 래퍼.

외부 dots.ocr 엔진 대신 Tesseract OCR(pytesseract)을 사용하여
이미지에서 텍스트와 바운딩 박스를 추출한다.

추출된 단어는 줄(line) 단위로 묶은 뒤 PyKoSpacing을 적용한다.
"""

from collections import defaultdict
from typing import Dict, List

try:  # 선택적 임포트: 미설치 시 명확한 에러를 발생
    from PIL import Image
    import pytesseract
    from pykospacing import Spacing
except Exception:  # pragma: no cover - optional dependency
    Image = None  # type: ignore
    pytesseract = None  # type: ignore
    Spacing = None  # type: ignore


class DotsOCR:
    """Tesseract 기반 간단 OCR 래퍼"""

    def __init__(self, **kwargs):
        self.opts = kwargs
        if Image is None or pytesseract is None or Spacing is None:
            raise ImportError(
                "pytesseract, Pillow, PyKoSpacing 패키지가 필요합니다"
            )

        # PyKoSpacing 객체는 비용이 크므로 한 번만 생성하여 재사용
        self.spacer = Spacing()

        # 하드코딩된 Tesseract 실행 파일 경로
        pytesseract.pytesseract.tesseract_cmd = (
            r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
        )

    def run(self, image_path: str) -> Dict:
        """이미지에서 OCR을 수행하고 줄 단위 결과를 반환"""

        img = Image.open(image_path)
        lang = self.opts.get("lang", "kor+eng")

        # 단어 단위 정보를 얻어오고 line_num 등 위치 정보를 함께 가져온다
        data = pytesseract.image_to_data(
            img, lang=lang, output_type=pytesseract.Output.DICT
        )

        groups: Dict[tuple, List[int]] = defaultdict(list)
        n = len(data["text"])
        for i in range(n):
            if not data["text"][i].strip():
                continue
            key = (
                data.get("block_num", [0])[i],
                data.get("par_num", [0])[i],
                data.get("line_num", [0])[i],
            )
            groups[key].append(i)

        blocks: List[Dict] = []
        for ln, idxs in enumerate(groups.values()):
            # 단어 정렬 (왼쪽→오른쪽)
            idxs = sorted(idxs, key=lambda j: data.get("word_num", [0])[j])
            words = [data["text"][j].strip() for j in idxs]
            raw = " ".join(words)
            # PyKoSpacing 적용
            spaced = self.spacer(raw)

            xs = [data["left"][j] for j in idxs]
            ys = [data["top"][j] for j in idxs]
            x2s = [data["left"][j] + data["width"][j] for j in idxs]
            y2s = [data["top"][j] + data["height"][j] for j in idxs]
            confs = []
            for j in idxs:
                cs = data["conf"][j]
                try:
                    confs.append(float(cs) / 100.0 if cs != "-1" else 0.0)
                except ValueError:
                    confs.append(0.0)

            blocks.append(
                {
                    "id": f"l{ln}",
                    "type": "paragraph",
                    "bbox": [min(xs), min(ys), max(x2s), max(y2s)],
                    "text": spaced,
                    "conf": sum(confs) / len(confs) if confs else 0.0,
                }
            )

        avg_conf = sum(b["conf"] for b in blocks) / len(blocks) if blocks else 0.0
        return {"page": 1, "avg_conf": avg_conf, "blocks": blocks, "lang": lang}