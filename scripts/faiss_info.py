#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, json, faiss

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="./data/index.faiss", help="FAISS index path")
    ap.add_argument("--meta", default=None, help="meta json path (default: <index>.faiss.meta.json)")
    args = ap.parse_args()

    index_path = args.index
    # ingest.py와 동일하게 <index>.faiss.meta.json 규칙 사용
    meta_path = args.meta or (index_path + ".meta.json")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta json not found: {meta_path}")

    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    print("=== FAISS INDEX INFO ===")
    print("index path:", index_path)
    print("ntotal:", index.ntotal)
    print("dim   :", index.d)
    print()
    print("=== META INFO ===")
    print("meta path:", meta_path)
    print("dim      :", meta.get("dim"))
    print("items    :", len(meta.get("items", [])))
    if meta.get("items"):
        it0 = meta["items"][0]
        print()
        print("=== SAMPLE ITEM[0] ===")
        print("text:", (it0.get("text","")[:200] + ("..." if len(it0.get("text",""))>200 else "")))
        print("meta:", it0.get("meta"))

if __name__ == "__main__":
    main()
