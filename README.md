# v-hnsw

Rust로 작성된 로컬 벡터 데이터베이스. HNSW 알고리즘 기반의 빠른 유사도 검색과 BM25 키워드 검색을 결합한 하이브리드 검색을 지원합니다.

## 주요 특징

- **올인원 CLI** - 데이터 추가, 검색, 관리를 단 2개 명령어로
- **자동 임베딩** - 텍스트를 자동으로 벡터로 변환 (100+ 언어 지원)
- **하이브리드 검색** - HNSW 벡터 검색 + BM25 키워드 검색 + RRF 융합
- **데몬 모드** - 모델을 메모리에 유지하여 검색 속도 23배 향상
- **GPU 가속** - CUDA / DirectML 지원
- **한국어 지원** - 형태소 분석 + 초성 검색

---

## 빠른 시작 (2단계)

```bash
# 1. 데이터 추가 (폴더 또는 파일)
v-hnsw add my-db ./documents/

# 2. 검색
v-hnsw find my-db "검색하고 싶은 내용"
```

끝! 모델 선택, DB 생성, 인덱싱 모두 자동으로 처리됩니다.

---

## 설치

### 원클릭 설치 (권장)

**Windows (PowerShell)**
```powershell
irm https://raw.githubusercontent.com/LeeSangMin1029/v-hnsw/main/install.ps1 | iex
```

**Linux / macOS**
```bash
curl -fsSL https://raw.githubusercontent.com/LeeSangMin1029/v-hnsw/main/install.sh | bash
```

### GitHub Releases

[Releases 페이지](https://github.com/LeeSangMin1029/v-hnsw/releases)에서 OS에 맞는 바이너리 다운로드

| 플랫폼 | CPU | CUDA GPU |
|--------|-----|----------|
| Windows x64 | `v-hnsw-windows.zip` | `v-hnsw-windows-cuda.zip` |
| Linux x64 | `v-hnsw-linux.tar.gz` | `v-hnsw-linux-cuda.tar.gz` |
| macOS | `v-hnsw-macos.tar.gz` | - |

### 소스에서 빌드

```bash
# CPU 버전
cargo build --release -p v-hnsw-cli

# CUDA GPU 버전
cargo build --release -p v-hnsw-cli --features cuda

# DirectML 버전 (Windows)
cargo build --release -p v-hnsw-cli --features directml
```

**요구 사항**: Rust 1.92+ (Edition 2024)

---

## 사용법

### 데이터 추가 (`add`)

```bash
# 마크다운 폴더 인덱싱
v-hnsw add my-db ./notes/

# JSONL 파일 인덱싱
v-hnsw add my-db data.jsonl

# Parquet 파일 인덱싱
v-hnsw add my-db data.parquet
```

**자동 처리 항목:**
- DB 생성 (없으면)
- 입력 타입 감지 (폴더/jsonl/parquet)
- 마크다운 시맨틱 청킹
- 임베딩 (multilingual-e5-base, 100+ 언어)
- HNSW + BM25 인덱스 빌드
- GPU 자동 감지 (CUDA 빌드 시)

### 검색 (`find`)

```bash
# 기본 검색
v-hnsw find my-db "API 보안 모범 사례"

# 결과 수 조절
v-hnsw find my-db "검색어" -k 20
```

**검색 방식**: 하이브리드 (HNSW 70% + BM25 30% + RRF 융합)

### 검색 출력 예시

```json
{
  "results": [
    {
      "id": 12345,
      "score": 0.89,
      "text": "API 보안을 위한 OWASP Top 10...",
      "source": "notes/security.md",
      "title": "API 보안 가이드"
    }
  ],
  "query": "API 보안 모범 사례",
  "elapsed_ms": 45
}
```

---

## 데몬 모드 (자동)

검색 시 백그라운드 데몬이 자동으로 시작되어 모델을 메모리에 유지합니다.

| 상태 | 검색 시간 |
|------|-----------|
| 데몬 없음 (첫 검색) | ~6초 |
| **데몬 사용 (이후)** | **~0.15초** |

- 첫 검색: 데몬 자동 시작 + 모델 로딩
- 이후 검색: 즉시 응답
- 5분 idle: 자동 종료 (메모리 반환)

```bash
# 수동으로 데몬 관리 (선택사항)
v-hnsw serve my-db          # 데몬 시작
v-hnsw serve my-db --stop   # 데몬 중지
```

---

## 고급 사용법

### 세부 옵션이 필요한 경우

```bash
# 데이터베이스 직접 생성
v-hnsw create my-db --dim 768 --metric cosine --neighbors 16 --ef 200

# 상세 옵션으로 삽입
v-hnsw insert my-db -i data.jsonl --embed --model bge-base-en-v1.5 --device cuda --batch-size 256

# 벡터 직접 검색
v-hnsw search my-db --vector "0.1,0.2,..." --k 10 --ef 500

# 텍스트로 검색 (BM25)
v-hnsw search my-db --text "키워드 검색" --k 10

# 시맨틱 검색 (쿼리 자동 임베딩)
v-hnsw vsearch my-db "검색 쿼리" --show-text
```

### 컬렉션 관리

```bash
v-hnsw collection my-db create my-collection --dim 384
v-hnsw collection my-db list
v-hnsw collection my-db delete my-collection
```

### 기타 명령어

```bash
v-hnsw info my-db                    # DB 정보
v-hnsw get my-db 42 43 44           # ID로 문서 조회
v-hnsw delete my-db --id 42         # 문서 삭제
v-hnsw build-index my-db            # 인덱스 재빌드
v-hnsw export my-db -o backup.jsonl # 내보내기
v-hnsw import my-db -i backup.jsonl # 가져오기
```

---

## 지원 임베딩 모델

| 모델 | 차원 | 크기 | 언어 | 비고 |
|------|------|------|------|------|
| **multilingual-e5-base** (기본) | 768 | 278MB | 100+ | 한국어 포함 |
| multilingual-e5-small | 384 | 118MB | 100+ | 경량 |
| multilingual-e5-large | 1024 | 560MB | 100+ | 고품질 |
| bge-base-en-v1.5 | 768 | 110MB | 영어 | 빠름 |
| bge-small-en-v1.5 | 384 | 33MB | 영어 | 경량 |
| all-mini-lm-l6-v2 | 384 | 22MB | 영어 | 최경량 |

---

## 지원 입력 포맷

| 포맷 | 확장자 | 설명 |
|------|--------|------|
| **마크다운 폴더** | `*.md` | 자동 시맨틱 청킹 |
| JSONL | `.jsonl`, `.ndjson` | `{"text": "...", "payload": {...}}` |
| Parquet | `.parquet` | 벡터/텍스트 컬럼 지정 가능 |
| fvecs/bvecs | `.fvecs`, `.bvecs` | 바이너리 벡터 포맷 |

### JSONL 형식

```json
{"text": "문서 내용", "payload": {"source": "file.md", "title": "제목"}}
```

---

## 프로젝트 구조

```
crates/
├── v-hnsw-core      # 핵심 트레잇, 타입
├── v-hnsw-distance  # SIMD 거리 함수 (L2, Cosine, Dot)
├── v-hnsw-graph     # HNSW 그래프 구현
├── v-hnsw-storage   # mmap 저장소 + WAL + 컬렉션
├── v-hnsw-search    # 하이브리드 검색 (HNSW + BM25 + RRF)
├── v-hnsw-embed     # 임베딩 모델 (fastembed)
├── v-hnsw-chunk     # 마크다운 시맨틱 청킹
├── v-hnsw-tokenizer # 한국어 형태소 분석 (Lindera)
└── v-hnsw-cli       # CLI 바이너리
```

---

## 성능

**환경**: RTX 3060 6GB, 1316 문서

| 작업 | 시간 |
|------|------|
| 인덱싱 (CUDA) | 35초 |
| 검색 (데몬) | 0.15초 |
| 검색 (직접) | 3.5초 |

---

## 라이선스

MIT OR Apache-2.0
