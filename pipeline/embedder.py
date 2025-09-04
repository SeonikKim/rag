# -*- coding: utf-8 -*-
from typing import List, Optional, Dict, Any
import os

class BaseEmbedder:
    def encode(self, texts: List[str]): raise NotImplementedError

# -------------------
# A) Qwen 임베딩
# -------------------
class QwenEmbedder(BaseEmbedder):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Embedding",
        normalize: bool = True,
        device: str = "cpu",
        batch_size: int = 16,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise ImportError(
                "sentence-transformers 임포트 실패. 'pip install \"transformers>=4.41,<5\" \"sentence-transformers>=2.7,<3\"' 로 설치했는지 확인하세요"
            ) from e

        self.model = SentenceTransformer(model_name, device=device)
        self.normalize = normalize
        self.dim = self.model.get_sentence_embedding_dimension()
        self.batch_size = batch_size

    def _l2(self, v):
        import numpy as np
        v = np.asarray(v, dtype="float32")
        return v / (np.linalg.norm(v) + 1e-12)

    def encode(self, texts: List[str]):
        embs = self.model.encode(
            texts,
            normalize_embeddings=False,
            show_progress_bar=False,
            convert_to_numpy=True,
            batch_size=self.batch_size,
        )
        if self.normalize:
            embs = [self._l2(v) for v in embs]
        else:
            embs = [v.astype("float32") for v in embs]
        return embs

# -------------------
# B) OpenAI 임베딩
# -------------------
class OpenAIEmbedder(BaseEmbedder):
    def __init__(
        self,
        model: str = "text-embedding-3-large",
        dim: int = 3072,
        normalize: bool = True,
        batch_size: int = 64,
    ):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY 환경변수가 필요합니다.")
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.dim = dim
        self.normalize = normalize
        self.batch_size = batch_size

    def _l2(self, v):
        import numpy as np
        v = np.asarray(v, dtype="float32")
        return v / (np.linalg.norm(v) + 1e-12)

    def encode(self, texts: List[str]):
        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            for e in resp.data:
                v = e.embedding
                out.append(self._l2(v) if self.normalize else v)
        return out

# -------------------
# C) Local 경량 SBERT
# -------------------
class LocalSBERTEmbedder(BaseEmbedder):
    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        dim: Optional[int] = None,
        normalize: bool = True,
        device: str = "cpu",
        batch_size: int = 16,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise ImportError(
                "sentence-transformers 임포트 실패. 'pip install \"transformers>=4.41,<5\" \"sentence-transformers>=2.7,<3\"' 로 설치했는지 확인하세요"
            ) from e

        self.model = SentenceTransformer(model_name, device=device)
        self.normalize = normalize
        self.dim = dim or self.model.get_sentence_embedding_dimension()
        self.batch_size = batch_size

    def _l2(self, v):
        import numpy as np
        v = np.asarray(v, dtype="float32")
        return v / (np.linalg.norm(v) + 1e-12)

    def encode(self, texts: List[str]):
        embs = self.model.encode(
            texts,
            normalize_embeddings=False,
            show_progress_bar=False,
            convert_to_numpy=True,
            batch_size=self.batch_size,
        )
        if self.normalize:
            embs = [self._l2(v) for v in embs]
        else:
            embs = [v.astype("float32") for v in embs]
        return embs

# -------------------
# Factory 함수
# -------------------
def get_embedder(cfg: Dict[str, Any]) -> BaseEmbedder:
    prov = (cfg.get("provider") or "qwen").lower()
    if prov == "qwen":
        return QwenEmbedder(
            model_name=cfg.get("model", "Qwen/Qwen2.5-Embedding"),
            normalize=cfg.get("normalize", True),
            device=cfg.get("device", "cpu"),
            batch_size=cfg.get("batch_size", 16),
        )
    elif prov == "openai":
        return OpenAIEmbedder(
            model=cfg.get("model", "text-embedding-3-large"),
            dim=cfg.get("dim", 3072),
            normalize=cfg.get("normalize", True),
            batch_size=cfg.get("batch_size", 64),
        )
    elif prov == "local":
        return LocalSBERTEmbedder(
            model_name=cfg.get("model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
            dim=cfg.get("dim"),
            normalize=cfg.get("normalize", True),
            device=cfg.get("device", "cpu"),
            batch_size=cfg.get("batch_size", 16),
        )
    else:
        raise ValueError(f"Unknown embeddings provider: {prov}")
