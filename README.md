# RAG Pipeline Skeleton (Python)
Created: 2025-09-03T01:22:08

## 구성
- `pipeline/pdf_to_image.py` : PDF → Image 변환 (DPI/전처리 훅)
- `pipeline/ocr_dots.py` : dots.ocr 호출 스텁(레이아웃+OCR 결과 스키마 유지)
- `pipeline/vision_fallback.py` : Varco/Kanana 멀티모달 대체 해석 스텁
- `pipeline/postprocess.py` : OCR 결과 정리(문단/표/리스트), 메타 유지
- `pipeline/exaone_struct.py` : Exaone 구조화/요약 스텁(Map-Reduce 훅)
- `pipeline/chunker.py` : 청킹 규칙(길이/타입/overlap)
- `pipeline/embedder.py` : 임베딩 스텁(bge-m3-ko 가정, dim 설정 가능)
- `pipeline/vector_sink.py` : VectorDB 업서트(JSON 파일 기본, Milvus/FAISS 훅)
- `ingest.py` : 전체 파이프라인 오케스트레이션(골격)
- `configs/config.yaml` : 파이프라인 파라미터
- `requirements.txt` : 의존성 목록(스텁 상태, 선택 설치)

## 빠른 실행 (스켈레톤 모드)
```bash
# 가상환경 권장
python ingest.py --pdf path/to/file.pdf --out ./out --config ./configs/config.yaml
```
> 주의: 현재 OCR/LLM/임베딩은 **스텁**입니다. 실제 엔진 연동 시 해당 파일들의 TODO 주석을 참고해 구현하세요.

## 교체 포인트
- dots.ocr 연동: `pipeline/ocr_dots.py` 의 `DotsOCR.run()`
- Varco/Kanana: `pipeline/vision_fallback.py` 의 `fallback_vision()`
- Exaone: `pipeline/exaone_struct.py` 의 `structure_and_summarize()`
- 임베딩 모델: `pipeline/embedder.py` 의 `Embedder`
- VectorDB: `pipeline/vector_sink.py` 의 `MilvusVectorSink`/`FaissVectorSink` TODO
- 하이브리드 검색: 별도 검색 서비스(Elasticsearch/OpenSearch) 연동 권장

## 라이선스
- 교육/내부 PoC 용 예시입니다. 실제 서비스용으로는 보안/로깅/예외/성능을 강화하세요.
