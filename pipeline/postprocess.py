# -*- coding: utf-8 -*-
"""
OCR/비전 해석 결과 → DocUnit[]
- 문단/표/리스트 등 의미 단위로 정리
- 메타데이터(page, source, conf, heading_path) 유지
"""
from typing import List, Dict
import re


def to_markdown_table(cells):
    # cells: List[List[{"text":str}]]
    if not cells:
        return ""
    header = "| " + " | ".join(c["text"] for c in cells[0]) + " |"
    sep = "| " + " | ".join("---" for _ in cells[0]) + " |"
    rows = ["| " + " | ".join(c["text"] for c in row) + " |" for row in cells[1:]]
    return "\n".join([header, sep] + rows)


CTRL_CHARS = re.compile(r"[\x00-\x1F\x7F]")
HEADER_FOOTER = re.compile(r"^(?:\d+|page\s*\d+|\d+\s*/\s*\d+)$", re.I)
TOC_LINE = re.compile(r"^\s*목차\s*$")


def normalize_text(text: str) -> str:
    text = CTRL_CHARS.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def infer_heading_path(page_json, block):
    # 제목 계층 추정을 위한 확장 포인트 (현재는 미사용)
    return []


def assemble_units_from_page(page_json: Dict, page_no: int, mode: str) -> List[Dict]:
    units: List[Dict] = []
    if mode == "ocr":
        blocks = page_json.get("blocks", [])
        avg_conf = page_json.get("avg_conf", 1.0)
        for b in sorted(blocks, key=lambda x: (x["bbox"][1], x["bbox"][0])):
            text = b.get("text", "").replace("-\n", "").replace("\n", " ").strip()
            if b.get("type") == "table":
                text = to_markdown_table(b.get("cells", []))
                typ = "table"
            else:
                typ = "paragraph" if b.get("type") != "title" else "title"
            units.append({
                "type": typ,
                "text": text,
                "page": page_no,
                "source": "ocr",
                "conf": b.get("conf", avg_conf),
                "heading_path": infer_heading_path(page_json, b)
            })
    elif mode == "pdf_text":
        raw = page_json.get("text", "")
        raw = raw.replace("-\n", "")
        lines = raw.splitlines()
        paras: List[str] = []
        buf: List[str] = []
        for ln in lines:
            s = ln.strip()
            if not s:

                if buf:
                    paras.append(" ".join(buf))
                    buf = []
                continue
            buf.append(s)
            # 빈 줄이 없어도 문장 종료 기호를 만나면 단락 분리
            if re.search(r"[\.?!]$|다\.$|요\.$", s):
                paras.append(" ".join(buf))
                buf = []
        if buf:
            paras.append(" ".join(buf))

        # 페이지 전체가 하나의 단락으로 뭉친 경우 추가 분리
        if len(paras) <= 1 and paras:
            sentences = re.split(r"(?<=[\.?!다요])\s+", paras[0])
            paras = []
            tmp: List[str] = []
            for sent in sentences:
                if not sent:
                    continue
                tmp.append(sent)
                if len(tmp) >= 2:
                    paras.append(" ".join(tmp))
                    tmp = []
            if tmp:
                paras.append(" ".join(tmp))


        h1 = None
        h2 = None
        h1_pat = re.compile(r"^[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\.")
        h2_pat = re.compile(r"^\d+\.")

        for p in paras:
            p = normalize_text(p)
            if not p or HEADER_FOOTER.match(p) or TOC_LINE.match(p):
                continue
            if h1_pat.match(p):
                h1 = p
                h2 = None
                continue
            if h2_pat.match(p):
                h2 = p
                continue
            heading_path = [x for x in [h1, h2] if x]
            if re.match(r"^[-\*•·]\s+", p):
                typ = "list_item"
                text = re.sub(r"^[-\*•·]\s+", "", p)
            elif "|" in p:
                typ = "table_row"
                text = p
            else:
                typ = "paragraph"
                text = p
            units.append({
                "type": typ,
                "text": text,
                "page": page_no,
                "source": "pdf_text",
                "conf": 1.0,
                "heading_path": heading_path,
            })
    else:
        # vision fallback
        summaries = page_json.get("summaries", [])
        facts = page_json.get("facts", [])
        triples = page_json.get("triples", [])
        if summaries:
            units.append({
                "type": "figure_summary",
                "text": " ".join(summaries),
                "page": page_no,
                "source": "vision_infer",
                "conf": 0.6,
                "heading_path": []
            })
        if facts or triples:
            units.append({
                "type": "figure_facts",
                "text": f"facts={facts}; triples={triples}",
                "page": page_no,
                "source": "vision_infer",
                "conf": 0.6,
                "heading_path": []
            })
    return units
