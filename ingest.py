#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, yaml, sys, traceback
from typing import List, Dict

from pipeline.pdf_to_image import pdf_to_images
from pipeline.ocr_dots import DotsOCR
from pipeline.vision_fallback import fallback_vision
from pipeline.postprocess import assemble_units_from_page
from pipeline.exaone_struct import structure_and_summarize
from pipeline.chunker import split_into_chunks
from pipeline.embedder import get_embedder
from pipeline.vector_sink import JSONVectorSink, FaissVectorSink


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def choose_sink(cfg: dict):
    vcfg = cfg.get("vector_sink", {})
    typ = vcfg.get("type", "faiss")
    if typ == "json":
        return JSONVectorSink(vcfg.get("json_path", "./data/index.json"))
    elif typ == "faiss":
        fc = vcfg.get("faiss", {})
        # metric 추가 지원 (L2, IP 등)
        metric = fc.get("metric", "L2").upper()
        if metric not in ("L2", "IP"):
            print(f"[WARN] Unknown FAISS metric={metric}, fallback=L2")
            fc["metric"] = "L2"
        return FaissVectorSink(fc)
    else:
        raise ValueError(f"Unknown vector_sink type: {typ}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="Input PDF path")
    ap.add_argument("--out", default="./out", help="Output directory for images")
    ap.add_argument("--config", default="./configs/config.yaml", help="YAML config path")
    args = ap.parse_args()

    try:
        cfg = load_config(args.config)
    except Exception as e:
        print(f"[ERROR] Config load failed: {e}")
        sys.exit(1)

    dpi = cfg["pipeline"]["dpi"]
    thr = cfg["pipeline"]["ocr_conf_threshold"]
    os.makedirs(args.out, exist_ok=True)

    # 1) PDF → Images
    print("[INFO] Step 1: PDF → Images")
    try:
        pages = pdf_to_images(args.pdf, dpi=dpi, out_dir=args.out, grayscale=True)
    except Exception as e:
        print(f"[ERROR] PDF to image failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 2) OCR or Fallback
    print("[INFO] Step 2: OCR → Units")
    ocr = DotsOCR()
    units: List[Dict] = []
    for page_meta in pages:
        try:
            ocr_page = ocr.run(page_meta["path"])
        except Exception as e:
            print(f"[ERROR] OCR failed on {page_meta['path']}: {e}")
            ocr_page = {"blocks": [], "avg_conf": 0.0}

        use_fallback = (
            ocr_page.get("avg_conf", 0.0) < thr
            or len(ocr_page.get("blocks", [])) == 0
        )
        if use_fallback:
            print(f"[WARN] Low OCR conf or empty, fallback: {page_meta['path']}")
            vf = fallback_vision(page_meta["path"])
            units.extend(assemble_units_from_page(vf, page_no=page_meta["page"], mode="vision"))
        else:
            units.extend(assemble_units_from_page(ocr_page, page_no=page_meta["page"], mode="ocr"))

    # 3) Exaone 구조화/요약
    print("[INFO] Step 3: Structure & Summarize")
    try:
        units = structure_and_summarize(units)
    except Exception as e:
        print(f"[ERROR] structure_and_summarize failed: {e}")

    # 4) 청킹
    print("[INFO] Step 4: Chunking")
    ccfg = cfg["chunk"]
    chunks = split_into_chunks(
        units,
        max_chars=ccfg["max_chars"],
        min_chars=ccfg["min_chars"],
        overlap_chars=ccfg["overlap_chars"],
    )

    # 5) 임베딩
    print("[INFO] Step 5: Embedding")
    ecfg = cfg["embedder"]
    embedder = get_embedder(ecfg)
    try:
        vectors = embedder.encode([c["text"] for c in chunks])
    except Exception as e:
        print(f"[ERROR] Embedding failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    assert len(chunks) == len(vectors), f"❌ chunks({len(chunks)}) != vectors({len(vectors)})"

    # 6) Vector Sink 업서트
    print("[INFO] Step 6: Vector Upsert")
    sink = choose_sink(cfg)
    try:
        sink.upsert(chunks, vectors)
    except Exception as e:
        print(f"[ERROR] Vector sink upsert failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    print(f"[OK] Ingested {len(pages)} pages → {len(units)} units → {len(chunks)} chunks")


if __name__ == "__main__":
    main()
