#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, yaml
from pipeline.embedder import get_embedder
from pipeline.vector_sink import FaissVectorSink

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, help="검색할 문장")
    ap.add_argument("--config", default="./configs/config.yaml", help="설정 파일 경로")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    emb = get_embedder(cfg["embedder"])

    # 질의 문장을 벡터로 변환
    qv = emb.encode([args.query])

    # FAISS 인덱스 로드 후 검색 수행
    vcfg = cfg["vector_sink"]["faiss"]
    sink = FaissVectorSink(vcfg)
    D, I = sink.search(qv, k=5)
    items = sink.meta["items"]
    for rank, (idx, score) in enumerate(zip(I[0], D[0]), start=1):
        if idx < 0 or idx >= len(items):
            continue
        it = items[idx]
        print(
            f"[{rank}] 점수={float(score):.4f} 페이지={it['meta'].get('pages')} 제목경로={it['meta'].get('heading_path')}"
        )
        t = it["text"].replace("\n", " ")
        print(t[:300] + ("..." if len(t) > 300 else ""))
        print("-" * 80)

if __name__ == "__main__":
    main()
