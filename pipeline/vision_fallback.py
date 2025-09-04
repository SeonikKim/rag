# -*- coding: utf-8 -*-
"""
Varco/Kanana 멀티모달 대체 해석 스텁
- OCR conf가 낮거나 표/그래프 중심 페이지에서 호출
- 이미지 내부 정보만 근거로 설명/추출
"""
from typing import Dict, List

def fallback_vision(image_path: str) -> Dict:
    # TODO: Varco/Kanana 호출 및 결과 정규화
    # ⚠️ OCR 본문 인덱싱에 합치지 말고 별도 저장소/필드로만 사용
    return {
        "mode": "vision_infer",
        "summaries": [],  # 노이즈 줄이기 위해 기본값은 비워둠
        "facts": [],      # 필요시 외부 모듈에서 채움
        "triples": [],
        "captions": []
    }