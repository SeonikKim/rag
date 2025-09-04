#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tesseract OCR 결과를 줄 단위로 묶고 PyKoSpacing 적용"""

from collections import defaultdict
from typing import Dict, List

try:  # 선택적 임포트: 설치되어 있지 않으면 None 으로 둔다
    from PIL import Image
    import pytesseract
    from pykospacing import Spacing
except Exception:  # pragma: no cover - optional dependency
    Image = None  # type: ignore
    pytesseract = None  # type: ignore
    Spacing = None  # type: ignore


def data_to_blocks(data: Dict, spacer) -> Dict:
    """Tesseract OCR 딕셔너리를 줄 단위 블록으로 변환"""

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
        # 단어를 왼쪽→오른쪽 순으로 정렬
        idxs = sorted(idxs, key=lambda j: data.get("word_num", [0])[j])
        words = [data["text"][j].strip() for j in idxs]
        raw = " ".join(words)
        spaced = spacer(raw) if spacer else raw

        xs = [data["left"][j] for j in idxs]
        ys = [data["top"][j] for j in idxs]
        x2s = [data["left"][j] + data["width"][j] for j in idxs]
        y2s = [data["top"][j] + data["height"][j] for j in idxs]
        confs: List[float] = []
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
    return {"blocks": blocks, "avg_conf": avg_conf}


class DotsOCR:
    """Tesseract 기반 간단 OCR 래퍼"""

    def __init__(self, **kwargs):
        self.opts = kwargs
        if Image is None or pytesseract is None or Spacing is None:
            raise ImportError(
                "pytesseract, Pillow, PyKoSpacing 패키지가 필요합니다",
            )

        # PyKoSpacing 객체는 비용이 크므로 한 번만 생성
        self.spacer = Spacing()

        # 하드코딩된 Tesseract 실행 파일 경로
        pytesseract.pytesseract.tesseract_cmd = (
            r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
        )

    def run(self, image_path: str) -> Dict:
        """이미지에서 OCR을 수행하고 줄 단위 결과를 반환"""

        img = Image.open(image_path)
        lang = self.opts.get("lang", "kor+eng")

        # 단어 단위 정보를 얻어온 뒤 data_to_blocks 로 처리
        data = pytesseract.image_to_data(
            img, lang=lang, output_type=pytesseract.Output.DICT
        )

        res = data_to_blocks(data, self.spacer)
        res.update({"page": 1, "lang": lang})
        return res

