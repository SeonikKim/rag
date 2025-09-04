# -*- coding: utf-8 -*-
"""
청킹(Chunking) 규칙: 길이 기반 + 타입 감지 + overlap
"""
from typing import List, Dict
import hashlib


def split_into_chunks(
    units: List[Dict],
    max_chars: int = 800,
    min_chars: int = 300,
    overlap_chars: int = 80,
) -> List[Dict]:
    chunks: List[Dict] = []
    buf: List[Dict] = []
    cur = 0

    def flush():
        nonlocal chunks, buf, cur
        if not buf:
            return
        text = "\n\n".join(x["text"] for x in buf if x.get("text"))
        # block_type은 첫 unit 기준으로 동일 타입이면 그 타입, 아니면 mixed
        types = {x.get("type", "paragraph") for x in buf}
        block_type = types.pop() if len(types) == 1 else "mixed"
        meta = {
            "pages": sorted(list({x.get("page") for x in buf})),
            "heading_path": buf[0].get("heading_path", []) if buf else [],
            "source": buf[0].get("source", "ocr") if buf else "ocr",
            "block_type": block_type,
        }
        chunks.append({"text": text, "meta": meta})
        buf.clear()
        cur = 0

    for u in units:
        tlen = len(u.get("text", ""))
        # 표/코드/제목은 독립 청크로 다루는 것이 안전
        if u.get("type") in ("table", "code", "title"):
            flush()
            chunks.append(
                {
                    "text": u.get("text", ""),
                    "meta": {
                        "pages": [u.get("page")],
                        "heading_path": u.get("heading_path", []),
                        "source": u.get("source", "ocr"),
                        "block_type": u.get("type"),
                    },
                }
            )
            continue

        if cur + tlen > max_chars and cur >= min_chars:
            flush()
        buf.append(u)
        cur += tlen

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

    # 내용 중복 제거
    seen = set()
    uniq_chunks: List[Dict] = []
    for c in chunks:
        h = hashlib.md5(c["text"].encode("utf-8")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        uniq_chunks.append(c)
    chunks = uniq_chunks

    # ID 부여
    for i, c in enumerate(chunks, start=1):
        c["id"] = f"chunk-{i:06d}"
    return chunks
