#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, yaml
from pipeline.embedder import get_embedder
from pipeline.vector_sink import FaissVectorSink

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--config", default="./configs/config.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config,"r",encoding="utf-8"))
    emb = get_embedder(cfg["embedder"])

    # embed query
    qv = emb.encode([args.query])

    # search
    vcfg = cfg["vector_sink"]["faiss"]
    sink = FaissVectorSink(vcfg)
    hits = sink._index.search(__import__("numpy").asarray(qv, dtype="float32"), 5)
    D, I = hits
    items = sink._meta["items"]
    for rank, (idx, score) in enumerate(zip(I[0], D[0]), start=1):
        if idx < 0 or idx >= len(items): continue
        it = items[idx]
        print(f"[{rank}] score={float(score):.4f} pages={it['meta'].get('pages')} heading={it['meta'].get('heading_path')}")
        t = it["text"].replace("\n"," ")
        print(t[:300] + ("..." if len(t)>300 else ""))
        print("-"*80)

if __name__ == "__main__":
    main()
