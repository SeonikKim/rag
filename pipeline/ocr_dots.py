# -*- coding: utf-8 -*-
"""
dots.ocr 연동 스텁
- 실제 엔진(REST/SDK) 호출 지점만 정의
- 출력 스키마는 레이아웃+텍스트+bbox+conf를 포함
"""
from typing import Dict

class DotsOCR:
    def __init__(self, **kwargs):
        self.opts = kwargs

    def run(self, image_path: str) -> Dict:
        # TODO: dots.ocr 호출(REST/SDK). 아래는 데모 스텁.
        return {
            "page": 1,
            "avg_conf": 0.90,
            "blocks": [
                {"id": "b1", "type": "title", "bbox": [50, 50, 500, 120], "text": "개요", "conf": 0.98},
                {"id": "b2", "type": "paragraph", "bbox": [50, 140, 1000, 400],
                 "text": "이 문서는 RAG 파이프라인의 개요를 설명한다.", "conf": 0.94},
                {"id": "t1", "type": "table", "bbox": [50, 420, 1000, 700],
                 "cells": [[{"text": "항목"}, {"text": "값"}],
                           [{"text": "연도"}, {"text": "2025"}]], "conf": 0.90}
            ],
            "lang": "ko"
        }
