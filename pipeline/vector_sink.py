# -*- coding: utf-8 -*-
"""
벡터 저장소(Vector Sink)
- JSON 파일 : 의존성 없이 간단히 확인 가능
- FAISS : CPU 기반 벡터 검색 지원
"""

import os
import json
import hashlib
from typing import List, Dict, Sequence


class JSONVectorSink:
    """JSON 파일에 벡터를 저장"""

    def __init__(self, path: str = "./data/index.json"):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not os.path.exists(self.path):
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({"meta": "rag-index", "items": []}, f, ensure_ascii=False, indent=2)

    def _load(self):
        if not os.path.exists(self.path):
            return {"meta": "rag-index", "items": []}
        with open(self.path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {"meta": "rag-index", "items": []}

    def _save(self, data):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def upsert(self, chunks: List[Dict], vectors: List[Sequence[float]]):
        data = self._load()
        items = data.get("items", [])
        for c, v in zip(chunks, vectors):
            vec = v.tolist() if hasattr(v, "tolist") else list(v)
            doc_id = c.get("meta", {}).get("doc_id", "unknown")
            key_src = f"{doc_id}-{c.get('id')}"
            uid = hashlib.md5(key_src.encode("utf-8")).hexdigest()
            items.append(
                {
                    "id": uid,
                    "chunk_id": c.get("id"),
                    "text": c.get("text"),
                    "vector": vec,
                    "meta": c.get("meta", {}),
                }
            )
        data["items"] = items
        self._save(data)


class MilvusVectorSink:
    """Milvus 스텁"""

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        # TODO: pymilvus 연동 구현

    def upsert(self, chunks: List[Dict], vectors: List[List[float]]):
        raise NotImplementedError("MilvusVectorSink: TODO - Implement with pymilvus")


class FaissVectorSink:
    """FAISS 기반 벡터 저장/검색"""

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.index_path = cfg.get("index_path", "./data/index.faiss")
        self.metric = cfg.get("metric", "L2").upper()
        self.meta_path = cfg.get("meta_path", self.index_path + ".meta.json")
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

        try:
            import faiss  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError("faiss 패키지가 필요합니다") from e
        self.faiss = faiss

        if os.path.exists(self.index_path):
            self.index = self.faiss.read_index(self.index_path)
        else:
            self.index = None

        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
        else:
            self.meta = {"items": []}

    def _create_index(self, dim: int):
        if self.metric == "IP":
            return self.faiss.IndexFlatIP(dim)
        else:
            return self.faiss.IndexFlatL2(dim)

    def upsert(self, chunks: List[Dict], vectors: List[Sequence[float]]):
        import numpy as np

        if not vectors:
            return

        vecs = np.array(vectors, dtype="float32")
        if self.index is None:
            self.index = self._create_index(vecs.shape[1])

        self.index.add(vecs)

        for c in chunks:
            doc_id = c.get("meta", {}).get("doc_id", "unknown")
            key_src = f"{doc_id}-{c.get('id')}"
            uid = hashlib.md5(key_src.encode("utf-8")).hexdigest()
            self.meta["items"].append(
                {
                    "id": uid,
                    "chunk_id": c.get("id"),
                    "text": c.get("text"),
                    "meta": c.get("meta", {}),
                }
            )

        self.faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)

    def search(self, vectors: List[Sequence[float]], k: int = 5):
        """주어진 벡터에 대해 FAISS 검색 수행"""
        import numpy as np

        if self.index is None:
            if os.path.exists(self.index_path):
                self.index = self.faiss.read_index(self.index_path)
            else:
                raise RuntimeError("FAISS 인덱스가 존재하지 않습니다")

        q = np.array(vectors, dtype="float32")
        return self.index.search(q, k)
