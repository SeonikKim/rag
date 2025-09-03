# -*- coding: utf-8 -*-
"""
OCR/비전 해석 결과 → DocUnit[]
- 문단/표/리스트 등 의미 단위로 정리
- 메타데이터(page, source, conf, heading_path) 유지
"""
from typing import List, Dict

def to_markdown_table(cells):
    # cells: List[List[{"text":str}]]
    if not cells:
        return ""
    header = "| " + " | ".join(c["text"] for c in cells[0]) + " |"
    sep = "| " + " | ".join("---" for _ in cells[0]) + " |"
    rows = ["| " + " | ".join(c["text"] for c in row) + " |" for row in cells[1:]]
    return "\n".join([header, sep] + rows)

def infer_heading_path(page_json, block):
    # TODO: 제목 계층 추정 로직(폰트/크기/번호 패턴 활용)
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
