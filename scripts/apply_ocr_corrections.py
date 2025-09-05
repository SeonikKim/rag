#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OCR로 추출한 텍스트를 사람이 수정한 후 units.json에 반영하는 스크립트"""
import argparse
import os
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./out", help="인제스트 결과가 있는 디렉터리")
    ap.add_argument(
        "--units", default="units.json", help="기존 유닛 JSON 파일 이름"
    )
    args = ap.parse_args()

    # units.json 로드
    units_path = os.path.join(args.out, args.units)
    with open(units_path, encoding="utf-8") as f:
        units = json.load(f)

    # 페이지별 OCR 유닛 목록 구성
    pages = {}
    for u in units:
        if u.get("source") == "ocr":
            pages.setdefault(u["page"], []).append(u)

    for page, ulist in pages.items():
        txt_path = os.path.join(args.out, f"p{page:04d}.txt")
        if not os.path.exists(txt_path):
            # 수정용 텍스트 파일이 없으면 건너뜀
            continue
        with open(txt_path, encoding="utf-8") as f:
            lines = [ln.rstrip("\n") for ln in f]
        if len(lines) != len(ulist):
            # 줄 수가 맞지 않으면 경고 후 건너뜀
            print(
                f"[WARN] 페이지 {page}: 줄 수 불일치 (units {len(ulist)} vs text {len(lines)})"
            )
            continue
        # 사용자 수정 내용을 units에 반영
        for u, line in zip(ulist, lines):
            u["text"] = line

    # 결과 저장
    out_path = os.path.join(args.out, "units_corrected.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(units, f, ensure_ascii=False, indent=2)
    print(f"[OK] 저장 완료: {out_path}")


if __name__ == "__main__":
    main()
