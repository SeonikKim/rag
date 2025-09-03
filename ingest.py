#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end ingest pipeline (skeleton)
- PDF → images
- dots.ocr → layout+text OR fallback vision
- Postprocess → Units
- Exaone structure (stub)
- Chunk → Embed → Vector upsert
"""
import argparse, os, sys, json, pathlib, yaml
from typing import List, Dict

from pipeline.pdf_to_image import pdf_to_images
from pipeline.ocr_dots import DotsOCR
from pipeline.vision_fallback import fallback_vision
from pipeline.postprocess import assemble_units_from_page
from pipeline.exaone_struct import structure_and_summarize
from pipeline.chunker import split_into_chunks
from pipeline.embedder import Embedder
from pipeline.vector_sink import JSONVectorSink, MilvusVectorSink, FaissVectorSink

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def choose_sink(cfg: dict):
    vcfg = cfg.get("vector_sink", {})
    typ = vcfg.get("type", "json")
    if typ == "json":
        return JSONVectorSink(vcfg.get("json_path", "./data/index.json"))
    elif typ == "milvus":
        return MilvusVectorSink(vcfg.get("milvus", {}))
    elif typ == "faiss":
        return FaissVectorSink(vcfg.get("faiss", {}))
    else:
        raise ValueError(f"Unknown vector_sink type: {typ}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="Input PDF path")
    ap.add_argument("--out", default="./out", help="Output directory (images, logs, etc.)")
    ap.add_argument("--config", default="./configs/config.yaml", help="YAML config path")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dpi = cfg["pipeline"]["dpi"]
    ocr_thr = cfg["pipeline"]["ocr_conf_threshold"]
    os.makedirs(args.out, exist_ok=True)

    # 1) PDF -> Images
    pages = pdf_to_images(args.pdf, dpi=dpi, out_dir=args.out)

    # 2) OCR / Fallback
    ocr = DotsOCR()
    all_units: List[Dict] = []
    for i, page_meta in enumerate(pages, start=1):
        img_path = page_meta["path"]
        ocr_page = ocr.run(img_path)
        use_fallback = (ocr_page.get("avg_conf", 0.0) < ocr_thr) or (len(ocr_page.get("blocks", [])) == 0)
        if use_fallback:
            vf = fallback_vision(img_path)
            units = assemble_units_from_page(vf, page_no=page_meta["page"], mode="vision")
        else:
            units = assemble_units_from_page(ocr_page, page_no=page_meta["page"], mode="ocr")
        all_units.extend(units)

    # 3) Exaone structure (stub)
    structured_units = structure_and_summarize(all_units)

    # 4) Chunk
    ccfg = cfg["chunk"]
    chunks = split_into_chunks(structured_units,
                               max_chars=ccfg["max_chars"],
                               min_chars=ccfg["min_chars"],
                               overlap_chars=ccfg["overlap_chars"])

    # 5) Embed
    ecfg = cfg["embedder"]
    embedder = Embedder(model=ecfg["model"], dim=ecfg["dim"])
    vectors = embedder.encode([c["text"] for c in chunks])

    # 6) Vector sink
    sink = choose_sink(cfg)
    sink.upsert(chunks, vectors)

    # Summary
    print(f"[OK] Ingested {len(pages)} pages → {len(structured_units)} units → {len(chunks)} chunks")

if __name__ == "__main__":
    main()
