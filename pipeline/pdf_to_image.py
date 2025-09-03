# -*- coding: utf-8 -*-
"""
PDF → Image 변환 (skeleton)
- PyMuPDF/fitz 사용 권장
- 전처리 훅(deskew/denoise/binarize) 제공 지점 표시
"""
import os, pathlib

def pdf_to_images(pdf_path: str, dpi: int = 300, out_dir: str = "./out", fmt: str = "png"):
    """
    Returns: List[{"page":int,"path":str,"dpi":int,"w":int,"h":int,"colorspace":"gray|rgb"}]
    """
    os.makedirs(out_dir, exist_ok=True)
    # NOTE: 실제 구현 시 아래를 PyMuPDF로 교체하세요.
    # 여기서는 스켈레톤으로 더미 이미지를 생성하지 않고 경로만 기록합니다.
    # 예시: p0001.png, p0002.png, ...
    # TODO: fitz.open(pdf_path) → page.get_pixmap(matrix=mat).save(...)
    # For now, assume 1-page placeholder.
    stub_path = os.path.join(out_dir, "p0001.png")
    # 사용자가 실제로 채워넣을 위치를 명확히 표시
    open(stub_path, "wb").write(b"")  # placeholder empty file
    return [{"page": 1, "path": stub_path, "dpi": dpi, "w": 0, "h": 0, "colorspace": "gray"}]
