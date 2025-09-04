#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, json, faiss

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="./data/index.faiss", help="FAISS 인덱스 경로")
    ap.add_argument(
        "--meta",
        default=None,
        help="메타 JSON 경로 (기본값: <index>.faiss.meta.json)",
    )

    args = ap.parse_args()

    index_path = args.index
    # ingest.py와 동일하게 <index>.faiss.meta.json 규칙 사용
    meta_path = args.meta or (index_path + ".meta.json")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS 인덱스를 찾을 수 없습니다: {index_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"메타 JSON을 찾을 수 없습니다: {meta_path}")

    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    print("=== FAISS 인덱스 정보 ===")
    print("인덱스 경로:", index_path)
    print("벡터 수(ntotal):", index.ntotal)
    print("차원(dim):", index.d)
    print()
    print("=== 메타 정보 ===")
    print("메타 경로:", meta_path)
    print("차원:", meta.get("dim"))
    print("아이템 수:", len(meta.get("items", [])))
    if meta.get("items"):
        it0 = meta["items"][0]
        print()
        print("=== 첫 번째 아이템 예시 ===")
        print("텍스트:", (it0.get("text","")[:200] + ("..." if len(it0.get("text",""))>200 else "")))
        print("메타:", it0.get("meta"))

if __name__ == "__main__":
    main()
