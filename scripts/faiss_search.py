#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, json, yaml, faiss, sys
import numpy as np
from sentence_transformers import SentenceTransformer


def load_cfg(path="./configs/config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_meta_path(index_path: str) -> str:
    """ingest.py와 동일한 규칙(<index>.faiss.meta.json)으로 메타 경로 계산"""
    return index_path + ".meta.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, help="검색할 문장")
    ap.add_argument("--config", default="./configs/config.yaml", help="설정 파일 경로")
    ap.add_argument("--k", type=int, default=5, help="상위 몇 개를 볼지")
    ap.add_argument("--device", default="cpu", help="SentenceTransformer 실행 장치(cpu/cuda)")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    vcfg = cfg["vector_sink"]["faiss"]
    index_path = vcfg.get("index_path", "./data/index.faiss")
    meta_path = compute_meta_path(index_path)

    if not os.path.exists(index_path):
        print(f"[ERROR] FAISS 인덱스를 찾을 수 없습니다: {index_path}\n→ 먼저 ingest.py로 색인하세요.")
        sys.exit(1)
    if not os.path.exists(meta_path):
        print(f"[ERROR] 메타 JSON을 찾을 수 없습니다: {meta_path}")
        sys.exit(1)

    # 1) 인덱스/메타 로드
    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    items = meta.get("items", [])
    if index.ntotal == 0 or len(items) == 0:
        raise RuntimeError("인덱스가 비어 있습니다. (chunks=0)")

    # 2) 임베더 로드
    ecfg = cfg["embedder"]
    model_id = ecfg.get("model", "Qwen/Qwen3-Embedding-0.6B")
    print(f"[INFO] 임베더 로드: {model_id} (장치={args.device})")
    model = SentenceTransformer(model_id, device=args.device)

    # 3) 질의 문장을 임베딩하고 검색
    qvec = model.encode([args.query], normalize_embeddings=True)
    qvec = np.asarray(qvec, dtype="float32")
    D, I = index.search(qvec, k=args.k)

    # 4) 결과 출력
    print(f"\n=== 검색 결과 상위 {args.k}개 ===")

    query_terms = args.query.split()
    for rank, (idx, score) in enumerate(zip(I[0], D[0]), start=1):
        if idx < 0 or idx >= len(items):
            continue
        it = items[idx]
        text = it.get("text", "").replace("\n", " ")
        for qt in query_terms:
            if qt:
                text = text.replace(qt, f"[{qt}]")
        meta_info = it.get("meta", {})
        print(f"[{rank}] 점수={float(score):.4f} 페이지={meta_info.get('pages')}")

        print(text[:300] + ("..." if len(text) > 300 else ""))
        print("-" * 80)


if __name__ == "__main__":
    main()
