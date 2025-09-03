# -*- coding: utf-8 -*-
"""
Exaone 구조화/요약 스텁
- 실제로는 LLM 호출(Map-Reduce)로 섹션/소제목/표/용어집/요약을 생성
- 스켈레톤에서는 입력을 그대로 반환(구조화 모드)하여 손실 방지
"""
from typing import List, Dict

def structure_and_summarize(units: List[Dict]) -> List[Dict]:
    # TODO: Exaone LLM 호출로 구조화·요약·용어집 생성
    # 현재는 구조화 손실을 피하기 위해 그대로 통과
    return units
