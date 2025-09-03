#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, json, yaml, faiss, sys
import numpy as np
from sentence_transformers import SentenceTransformer

def load_cfg(path="./configs/config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def compute_meta_path(index_path: str) -> str:
    base, _ = os.path.splitext(index_path)
    return base + ".meta.json"  # ./data/index.faiss -> ./data/index.meta.json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, help="검색 쿼리 문장")
    ap.add_argument("--config", default="./configs/config.yaml", help="YAML config path")
    ap.add_argument("--k", type=int, default=5, help="top-k")
    ap.add_argument("--device", default="cpu", help="SentenceTransformer device (cpu/cuda)")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    vcfg = cfg["vector_sink"]["faiss"]
    index_path = vcfg.get("index_path", "./data/index.faiss")
    meta_path = compute_meta_path(index_path)

    if not os.path.exists(index_path):
        print(f"[ERROR] FAISS index not found: {index_path}\n→ 먼저 ingest.py로 색인하세요.")
        sys.exit(1)
    if not os.path.exists(meta_path):
        print(f"[ERROR] Meta json not found: {meta_path}")
        sys.exit(1)

    # 1) 인덱스/메타 로드
    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    items = meta.get("items", [])
    if index.ntotal == 0 or len(items) == 0:
        raise RuntimeError("인덱스가 비어 있습니다. (chunks=0)")

    # 2) 임베더 로드 (config의 embedder.model 사용)
    ecfg = cfg["embedder"]
    model_id = ecfg.get("model", "Qwen/Qwen3-Embedding-0.6B")
    print(f"[INFO] Loading embedder: {model_id} (device={args.device})")
    model = SentenceTransformer(model_id, device=args.device)

    # 3) 쿼리 임베딩 → 검색
    qvec = model.encode([args.query], normalize_embeddings=True)
    qvec = np.asarray(qvec, dtype="float32")
    D, I = index.search(qvec, k=args.k)

    # 4) 출력 (안전 모드)
    print(f"\n=== RESULTS (top-{args.k}) ===")
    threshold = 0.55  # 유사도 기준 (실험 후 조정)
    found = False
    query_terms = args.query.split()