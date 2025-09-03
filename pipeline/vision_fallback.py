# -*- coding: utf-8 -*-
"""
Varco/Kanana 멀티모달 대체 해석 스텁
- OCR conf가 낮거나 표/그래프 중심 페이지에서 호출
- 이미지 내부 정보만 근거로 설명/추출
"""
from typing import Dict, List

def fallback_vision(image_path: str) -> Dict:
    # TODO: Varco/Kanana 호출 및 결과 정규화
    return {
        "mode": "vision_infer",
        "summaries": ["라인 차트로 2018~2024 매출 추이를 표현, 2024에 최고치"],
        "facts": [{"k": "x_axis", "v": "연도"}, {"k": "y_axis", "v": "매출(억원)"}],
        "triples": [["2021", "매출", "250"]],
        "captions": [{"bbox": [60, 420, 1000, 700], "text": "Figure 2: Sales Trend"}]
    }
