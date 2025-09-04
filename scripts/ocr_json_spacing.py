#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OCR JSON을 줄 단위로 묶고 PyKoSpacing 적용"""

import argparse, json

try:  # PyKoSpacing 미설치 시 사용자에게 안내
    from pykospacing import Spacing
except Exception:  # pragma: no cover - optional dependency
    Spacing = None  # type: ignore

from pipeline.ocr_dots import data_to_blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Tesseract image_to_data 결과 JSON 경로")
    ap.add_argument("--output", required=True, help="줄 단위로 재구성한 JSON 경로")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    if Spacing is None:
        raise ImportError("PyKoSpacing 패키지가 필요합니다")
    spacer = Spacing()
    res = data_to_blocks(data, spacer)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    print(f"[OK] {len(res['blocks'])}줄 저장: {args.output}")


if __name__ == "__main__":
    main()

