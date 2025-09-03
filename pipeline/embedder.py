# -*- coding: utf-8 -*-
"""
임베딩 스텁
- 실제 모델(bge-m3-ko 등) 연동 전까지는 0-벡터 반환
- 차원은 config.embedder.dim을 따름
"""
from typing import List
import math

class Embedder:
    def __init__(self, model: str = "bge-m3-ko", dim: int = 1024):
        self.model = model
        self.dim = dim

    def encode(self, texts: List[str]):
        # TODO: 실제 임베딩 모델 연동
        # 스텁: 텍스트 길이를 기반으로 간단한 해시성 값으로 채움(디버깅용)
        vecs = []
        for t in texts:
            base = (len(t) % 1000) / 1000.0
            v = [base * ((i % 13)+1) / 13.0 for i in range(self.dim)]
            vecs.append(v)
        return vecs
