# -*- coding: utf-8 -*-
"""
청킹(Chunking) 규칙: 길이 기반 + 타입 감지 + overlap
"""
from typing import List, Dict

def split_into_chunks(units: List[Dict], max_chars=1400, min_chars=800, overlap_chars=200) -> List[Dict]:
    chunks = []
    buf = []
    cur = 0

    def flush():
        nonlocal chunks, buf, cur
        if not buf:
            return
        text = "\n\n".join(x["text"] for x in buf if x.get("text"))
        meta = {
            "pages": sorted(list({x.get("page") for x in buf})),
            "heading_path": buf[0].get("heading_path", []) if buf else [],
            "source": buf[0].get("source", "ocr") if buf else "ocr"
        }
        chunks.append({"text": text, "meta": meta})
        buf.clear(); cur = 0

    for u in units:
        tlen = len(u.get("text", ""))
        # 표/코드/제목은 독립 청크로 다루는 것이 안전
        if u.get("type") in ("table", "code", "title"):
            flush()
            chunks.append({"text": u.get("text",""), "meta":{
                "pages":[u.get("page")], "heading_path": u.get("heading_path", []),
                "source": u.get("source","ocr")
            }})
            continue

        if cur + tlen > max_chars and cur >= min_chars:
            flush()
        buf.append(u); cur += tlen

    flush()

    # overlap 부여
    if overlap_chars > 0 and len(chunks) > 1:
        with_ov = []
        prev_tail = ""
        for c in chunks:
            text = c["text"]
            joined = (prev_tail + "\n\n" + text) if prev_tail else text
            with_ov.append({"text": joined, "meta": c["meta"]})
            prev_tail = text[-overlap_chars:]
        chunks = with_ov

    # ID 부여
    for i, c in enumerate(chunks, start=1):
        c["id"] = f"chunk-{i:06d}"
    return chunks
