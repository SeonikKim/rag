# RAG Pipeline Skeleton (Python)
Created: 2025-09-03T01:22:08

## 구성
- `pipeline/pdf_to_image.py` : PDF → Image 변환 (DPI/전처리 훅)
- `pipeline/ocr_dots.py` : dots.ocr 호출 스텁(레이아웃+OCR 결과 스키마 유지)
- `pipeline/vision_fallback.py` : Varco/Kanana 멀티모달 대체 해석 스텁
- `pipeline/postprocess.py` : OCR 결과 정리(문단/표/리스트), 메타 유지
- `pipeline/exaone_struct.py` : Exaone 구조화/요약 스텁(Map-Reduce 훅)
- `pipeline/chunker.py` : 청킹 규칙(길이/타입/overlap)
- `pipeline/embedder.py` : 임베딩 스텁(Qwen/OpenAI/경량 SBERT 지원)
- `pipeline/vector_sink.py` : VectorDB 업서트(JSON 파일 기본, Milvus/FAISS 훅)
- `ingest.py` : 전체 파이프라인 오케스트레이션(골격, PDF 내 텍스트 존재 시 OCR 생략)
- `configs/config.yaml` : 파이프라인 파라미터
- `requirements.txt` : 의존성 목록(스텁 상태, 선택 설치)
- `scripts/rechunk_meta.py` : 기존 meta.json을 문단·불릿·표 행 단위로 재청킹

## 빠른 실행 (스켈레톤 모드)
```bash
# 가상환경 권장
python ingest.py --pdf path/to/file.pdf --out ./out --config ./configs/config.yaml
# 모든 페이지를 강제로 OCR하려면 --ocr-only 옵션 추가
python ingest.py --pdf path/to/file.pdf --out ./out --config ./configs/config.yaml --ocr-only
```
> 주의: 현재 OCR/LLM/임베딩은 **스텁**입니다. 기본적으로 PDF에 텍스트가 포함되어 있으면 OCR 없이 처리하며, 스캔본 페이지에만 OCR이 동작합니다. 모든 페이지에 OCR을 적용하려면 `--ocr-only` 옵션을 사용하세요. 실제 엔진 연동 시 해당 파일들의 TODO 주석을 참고해 구현하세요.

## OCR 텍스트 검증 및 수정
- `ingest.py` 실행이 끝나면 OCR 처리된 각 페이지의 이미지와 텍스트가 자동으로 표시됩니다.
  - 환경변수 `EDITOR` 가 설정되어 있으면 해당 편집기가 열리고, 그렇지 않으면 경로만 안내합니다.
  - 텍스트를 저장하고 창을 닫으면 다음 페이지로 넘어갑니다.
- 사용자가 수정한 내용은 자동으로 `out/units_corrected.json`에 반영되며 이후 파이프라인에서 활용됩니다.

## OCR 텍스트 검증 및 수정
- `ingest.py` 실행이 끝나면 OCR 처리된 각 페이지의 이미지와 텍스트가 자동으로 표시됩니다.
  - 환경변수 `EDITOR` 가 설정되어 있으면 해당 편집기가 열리고, 그렇지 않으면 경로만 안내합니다.
  - 텍스트를 저장하고 창을 닫으면 다음 페이지로 넘어갑니다.
- 사용자가 수정한 내용은 자동으로 `out/units_corrected.json`에 반영되며 이후 파이프라인에서 활용됩니다.


## `meta.json` 재청킹
1. 인제스트가 완료되면 `.faiss.meta.json` 파일이 생성됩니다.
2. 해당 메타를 문단·불릿·표 행 단위로 재청킹하려면 다음 스크립트를 실행하세요.

### POSIX (Linux/macOS)
```bash
python scripts/rechunk_meta.py \
  --meta ./data/sample_index.faiss.meta.json \
  --out ./data/sample_index.rechunk.json \
  --max_chars 800 --min_chars 300 --overlap 80
```

### Windows PowerShell
```powershell
python scripts/rechunk_meta.py `
  --meta ./data/sample_index.faiss.meta.json `
  --out ./data/sample_index.rechunk.json `
  --max_chars 800 --min_chars 300 --overlap 80
```

### Windows CMD
```cmd
python scripts/rechunk_meta.py ^
  --meta .\data\sample_index.faiss.meta.json ^
  --out .\data\sample_index.rechunk.json ^
  --max_chars 800 --min_chars 300 --overlap 80
```
> Windows에서는 `\` 로 줄바꿈을 할 수 없으니 한 줄로 실행하거나 위와 같이 `^`(CMD), `` ` ``(PowerShell)을 사용하세요.

## 저사양(CPU) 환경 실행
- `configs/config.yaml` 은 기본적으로 GPU 없이 동작하도록 경량 SBERT 임베딩과 낮은 DPI(200)를 사용합니다.
- 8GB RAM 수준의 랩탑에서는 `batch_size` 나 `dpi` 값을 필요에 맞게 추가 조정할 수 있습니다.

## 교체 포인트
- dots.ocr 연동: `pipeline/ocr_dots.py` 의 `DotsOCR.run()`
- Varco/Kanana: `pipeline/vision_fallback.py` 의 `fallback_vision()`
- Exaone: `pipeline/exaone_struct.py` 의 `structure_and_summarize()`
- 임베딩 모델: `pipeline/embedder.py` 의 `Embedder`
- VectorDB: `pipeline/vector_sink.py` 의 `MilvusVectorSink`/`FaissVectorSink` TODO
- 하이브리드 검색: 별도 검색 서비스(Elasticsearch/OpenSearch) 연동 권장

## 라이선스
- 교육/내부 PoC 용 예시입니다. 실제 서비스용으로는 보안/로깅/예외/성능을 강화하세요.
