#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tesseract OCR 결과를 줄 단위로 묶고 PyKoSpacing 적용"""


from collections import defaultdict
from typing import Dict, List

try:  # 선택적 임포트: 설치되어 있지 않으면 None 으로 둔다
    from PIL import Image
    import pytesseract
    from pykospacing import Spacing
    import cv2
    import numpy as np

except Exception:  # pragma: no cover - optional dependency
    Image = None  # type: ignore
    pytesseract = None  # type: ignore
    Spacing = None  # type: ignore
    cv2 = None  # type: ignore
    np = None  # type: ignore



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
        raw = "".join(words)  # 단어 사이 공백 제거 후 스페이싱 적용
        spaced = spacer(raw) if spacer else raw
        spaced = spaced.replace("  ", " ")  # 이중 공백 정리


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


def local_polarity(gray: "np.ndarray", tile: int = 128) -> "np.ndarray":
    """타일 단위로 극성을 판정해 적응형 이진화를 적용"""

    h, w = gray.shape
    out = gray.copy()
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            ty = min(y + tile, h)
            tx = min(x + tile, w)
            tile_img = gray[y:ty, x:tx]
            mean = tile_img.mean()
            inv = mean < 128  # 어두운 배경이면 반전 사용
            ttype = cv2.THRESH_BINARY_INV if inv else cv2.THRESH_BINARY
            bin_tile = cv2.adaptiveThreshold(
                tile_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, ttype, 25, 15
            )
            out[y:ty, x:tx] = bin_tile
    return out



    avg_conf = sum(b["conf"] for b in blocks) / len(blocks) if blocks else 0.0
    return {"blocks": blocks, "avg_conf": avg_conf}

class DotsOCR:
    """Tesseract 기반 간단 OCR 래퍼"""

    def __init__(self, psm: int = 6, oem: int = 1, **kwargs):
        self.opts = kwargs
        self.psm = psm
        self.oem = oem
        if None in (Image, pytesseract, Spacing, cv2, np):
            raise ImportError(
                "pytesseract, Pillow, OpenCV, numpy, PyKoSpacing 패키지가 필요합니다",
            )

        # PyKoSpacing 객체는 비용이 크므로 한 번만 생성

        self.spacer = Spacing()

        # 하드코딩된 Tesseract 실행 파일 경로
        pytesseract.pytesseract.tesseract_cmd = (
            r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
        )

    def run(self, image_path: str) -> Dict:
        """여러 전처리 조합으로 OCR을 수행하고 최고 신뢰 결과 반환"""


        lang_pair = self.opts.get("lang_pair", ("kor+eng", "kor"))

        # 원본 이미지를 그레이스케일로 로드
        gray = Image.open(image_path).convert("L")
        gray_np = np.array(gray)

        # 공통 전처리 후보들을 미리 계산
        variants: List["np.ndarray"] = [gray_np]
        variants.append(cv2.bitwise_not(gray_np))
        th = cv2.adaptiveThreshold(
            gray_np, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 15
        )
        variants.append(th)
        variants.append(local_polarity(gray_np))

        def ocr_array(arr: "np.ndarray", lang: str) -> Dict:
            """넘파이 배열을 받아 OCR 수행"""
            pil = Image.fromarray(arr)
            data = pytesseract.image_to_data(
                pil,
                lang=lang,
                config=f"--psm {self.psm} --oem {self.oem}",
                output_type=pytesseract.Output.DICT,
            )
            return data_to_blocks(data, self.spacer)

        def best_for_lang(lang: str) -> Dict:
            """여러 전처리 중 평균 신뢰도 최고 결과 반환"""
            cands: List[Dict] = [ocr_array(v, lang) for v in variants]
            return max(cands, key=lambda d: d.get("avg_conf", 0.0))

        # 두 언어 설정으로 각각 최고 결과 산출
        res_mixed = best_for_lang(lang_pair[0])
        res_kor = best_for_lang(lang_pair[1])

        # 줄 단위로 신뢰도 비교하여 높은 쪽 채택
        blocks: List[Dict] = []
        max_len = max(len(res_mixed["blocks"]), len(res_kor["blocks"]))
        for i in range(max_len):
            b1 = res_mixed["blocks"][i] if i < len(res_mixed["blocks"]) else None
            b2 = res_kor["blocks"][i] if i < len(res_kor["blocks"]) else None
            if b1 and b2:
                blocks.append(b1 if b1["conf"] >= b2["conf"] else b2)
            elif b1:
                blocks.append(b1)
            elif b2:
                blocks.append(b2)

        avg_conf = sum(b["conf"] for b in blocks) / len(blocks) if blocks else 0.0
        return {"blocks": blocks, "avg_conf": avg_conf, "page": 1, "lang": "/".join(lang_pair)}


