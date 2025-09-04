#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
meta.json 재분할 및 heading_path 보강 스크립트
- 입력: 기존 chunk 메타(JSON 배열)
- 출력: 문단/불릿/표 행 단위로 재청킹된 JSON
"""
import argparse
import json
import os
import re
import sys
from typing import List, Dict

# repo root를 import 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pipeline.chunker import split_into_chunks

H1_PAT = re.compile(r"^[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\.")
H2_PAT = re.compile(r"^\d+\.")
BULLET_PAT = re.compile(r"^[-\*•·]\s+")


def iter_units(chunk: Dict) -> List[Dict]:
    """기존 청크 텍스트를 문단/불릿/표 행 단위 unit으로 분할"""
    text = chunk.get("text", "")
    meta = chunk.get("meta", {})
    pages = meta.get("pages", [])
    source = meta.get("source", "ocr")
    h1 = None
    h2 = None
    units: List[Dict] = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if H1_PAT.match(line):
            h1 = line
            h2 = None
            continue
        if H2_PAT.match(line):
            h2 = line
            continue
        heading_path = [x for x in [h1, h2] if x]
        if BULLET_PAT.match(line):
            typ = "list_item"
            line = BULLET_PAT.sub("", line)
        elif "|" in line:
            typ = "table_row"
        else:
            typ = "paragraph"
        units.append(
            {
                "type": typ,
                "text": line,
                "page": pages[0] if pages else None,
                "source": source,
                "heading_path": heading_path,
            }
        )
    return units


def process(meta_path: str, out_path: str, max_chars: int, min_chars: int, overlap: int):
    data = json.load(open(meta_path, "r", encoding="utf-8"))
    # FaissVectorSink 메타 파일({"items": [...]})과
    # 기존 배열 형태([{"text": ..., "meta": ...}, ...])를 모두 지원
    if isinstance(data, dict) and "items" in data:
        data = data["items"]

    all_units: List[Dict] = []
    for chunk in data:
        all_units.extend(iter_units(chunk))
    chunks = split_into_chunks(
        all_units,
        max_chars=max_chars,
        min_chars=min_chars,
        overlap_chars=overlap,
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Wrote {len(chunks)} chunks → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="기존 meta.json 경로")
    ap.add_argument("--out", default="meta_rechunk.json", help="출력 경로")
    ap.add_argument("--max_chars", type=int, default=800, help="청크 최대 길이")
    ap.add_argument("--min_chars", type=int, default=300, help="청크 최소 길이")
    ap.add_argument("--overlap", type=int, default=80, help="오버랩 문자 수")
    args = ap.parse_args()
    process(args.meta, args.out, args.max_chars, args.min_chars, args.overlap)


if __name__ == "__main__":
    main()