# -*- coding: utf-8 -*-
"""PDF → Image 변환 유틸리티.

PyMuPDF(fitz)를 이용해 PDF 페이지를 이미지로 렌더링한다.
`grayscale=True`이면 회색조 이미지로 변환하며, 반환 메타데이터에
사용된 색공간과 해상도, 이미지 크기 등을 기록한다.
"""

import os
from typing import Dict, List

import fitz  # PyMuPDF


def pdf_to_images(
    pdf_path: str,
    dpi: int = 300,
    out_dir: str = "./out",
    fmt: str = "png",
    grayscale: bool = False,
    page_numbers: List[int] | None = None,
) -> List[Dict]:
    """Convert a PDF into page images.

    Args:
        pdf_path: 입력 PDF 경로
        dpi: 렌더링 DPI (72가 100%)
        out_dir: 출력 디렉터리
        fmt: 저장 포맷 (png, jpg 등)
        grayscale: True면 회색조, False면 RGB로 저장
        page_numbers: 변환할 페이지 번호 리스트 (1-indexed). None이면 전체 페이지

    Returns:
        각 페이지에 대한 메타데이터 리스트. 페이지 번호, 이미지 경로, dpi,
        폭/높이, 색공간 정보를 포함한다.
    """
    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    images: List[Dict] = []

    try:
        selected = set(page_numbers) if page_numbers else None
        for page_no, page in enumerate(doc, start=1):
            if selected and page_no not in selected:
                continue
            # DPI 반영
            mat = fitz.Matrix(dpi / 72, dpi / 72)

            if grayscale:
                pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
                colorspace = "gray"
            else:
                pix = page.get_pixmap(matrix=mat)  # 기본 RGB
                colorspace = "rgb"

            fname = f"p{page_no:04d}.{fmt}"
            fpath = os.path.join(out_dir, fname)
            pix.save(fpath)

            images.append(
                {
                    "page": page_no,
                    "path": fpath,
                    "dpi": dpi,
                    "w": pix.width,
                    "h": pix.height,
                    "colorspace": colorspace,
                }
            )
    finally:
        doc.close()

    return images
