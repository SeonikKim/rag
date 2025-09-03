# -*- coding: utf-8 -*-
"""
Vector Sink
- JSON 파일(기본) : 의존성 없이 빠르게 확인 가능
- Milvus/FAISS : TODO 스텁(연동 지점 명시)
"""
import os
import json
import hashlib
from typing import List, Dict, Sequence


class JSONVectorSink:
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
            items.append({
                "id": uid,
                "chunk_id": c.get("id"),
                "text": c.get("text"),
                "vector": vec,
                "meta": c.get("meta", {}),
            })
        data["items"] = items
        self._save(data)


class MilvusVectorSink:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        # TODO: pymilvus 연결 및 스키마 생성

    def upsert(self, chunks: List[Dict], vectors: List[List[float]]):
        # TODO: upsert 구현
        raise NotImplementedError("MilvusVectorSink: TODO - Implement with pymilvus")


class FaissVectorSink:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.index_path = cfg.get("index_path", "./data/index.faiss")
        self.metric = cfg.get("metric", "L2").upper()
        self.meta_path = cfg.get("meta_path", self.index_path + ".meta.json")
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

        try:
            import faiss  # type: ignore
        except ImportError as e:
            raise ImportError("faiss package is required for FaissVectorSink") from e
        self.faiss = faiss

        if os.path.exists(self.index_path):
            self.index = self.faiss.read_index(self.index_path)
        else:
            self.index = None

        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
        else:
            self.meta = []

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
            self.meta.append(
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