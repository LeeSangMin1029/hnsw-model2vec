# v-hnsw

Rust로 작성된 로컬 벡터 데이터베이스. HNSW + BM25 하이브리드 검색 CLI.

## 설치

### Windows — 빌드 없이 바로 설치

```bash
git clone https://github.com/LeeSangMin1029/hnsw-model2vec.git v-hnsw
cd v-hnsw
install.bat
```

GitHub Release에서 빌드된 exe를 다운로드합니다. Rust, Visual Studio 등 추가 설치 불필요.

### Linux / macOS

```bash
git clone https://github.com/LeeSangMin1029/hnsw-model2vec.git v-hnsw
cd v-hnsw
chmod +x install.sh && ./install.sh
```

`gh` CLI가 있으면 Release에서 바이너리 다운로드, 없으면 소스 빌드 (Rust 미설치 시 자동 설치).

### 소스 빌드 (직접)

```bash
cargo build --release -p v-hnsw-cli
```

빌드된 바이너리: `target/release/v-hnsw` (Linux/macOS) / `target/release/v-hnsw.exe` (Windows)

### 첫 실행 시 자동 다운로드

첫 `add` 또는 `find` 실행 시 아래 항목이 자동 다운로드됩니다:

- **임베딩 모델** — `minishlab/potion-multilingual-128M` (~500MB, HuggingFace Hub)
- **한국어 사전** — `ko-dic` (Lindera, `~/.v-hnsw/dict/ko-dic/`)

이후에는 로컬 캐시를 사용합니다.

## 주요 기능

- **하이브리드 검색** — HNSW 벡터 + BM25 키워드 + Convex Fusion
- **자동 임베딩** — model2vec 기반 (100+ 언어, 256차원)
- **데몬 모드** — mmap 스냅샷 즉시 로딩, ~2ms 검색
- **한국어 지원** — Lindera ko-dic 형태소 분석
- **증분 업데이트** — 변경된 파일만 감지하여 재인덱싱
- **태그 필터링** — 마크다운 프론트매터 태그 기반 Roaring Bitmap 필터

## 사용법

```bash
# 1. 데이터 추가 (마크다운 폴더)
v-hnsw add my-db ./documents/

# 2. 검색
v-hnsw find my-db "검색하고 싶은 내용"

# 3. 변경된 파일만 업데이트
v-hnsw update my-db ./documents/
```

모델 다운로드, DB 생성, 인덱싱 모두 자동으로 처리됩니다.

### 서브커맨드

| 커맨드 | 설명 |
|--------|------|
| `add` | 마크다운/JSONL/Parquet 추가 (청킹+임베딩+인덱싱) |
| `find` | 하이브리드 검색 (데몬 자동 시작) |
| `update` | 변경 파일만 증분 업데이트 |
| `info` | DB 정보 (문서 수, 크기, 설정) |
| `get` | ID로 문서 조회 |
| `delete` | 문서 삭제 |
| `build-index` | 인덱스 재빌드 (스냅샷 포함) |
| `export` | JSONL 내보내기 |
| `import` | JSONL 가져오기 |
| `serve` | 데몬 수동 관리 |

### find — 검색

```bash
# 하이브리드 검색 (기본)
v-hnsw find my-db "API 보안 모범 사례"

# 결과 수 조절
v-hnsw find my-db "검색어" -k 20

# 태그 필터링 (AND 조건)
v-hnsw find my-db "검색어" --tag rust --tag architecture

# 전체 텍스트 표시
v-hnsw find my-db "검색어" --full
```

### add — 데이터 추가

```bash
# 마크다운 폴더
v-hnsw add my-db ./notes/

# JSONL 파일
v-hnsw add my-db data.jsonl

# Parquet 파일
v-hnsw add my-db data.parquet
```

입력 타입 자동 감지, 시맨틱 청킹, 임베딩, 인덱싱까지 한 번에 처리합니다.

### 검색 결과 예시

```json
{
  "results": [
    {
      "id": 12847765652211020711,
      "score": 1.0,
      "text": "API 보안을 위한 OWASP Top 10...",
      "source": "notes/security.md",
      "title": "API 보안 가이드"
    }
  ],
  "elapsed_ms": 2.2
}
```

## 태그 시스템

마크다운 프론트매터에 태그를 지정하면 검색 시 필터링할 수 있습니다.

```markdown
---
tags: [project-structure, rust, architecture]
title: 프로젝트 아키텍처
---

# 프로젝트 구조
...
```

```bash
v-hnsw find my-db "아키텍처" --tag project-structure
```

## 데몬 모드

`find` 실행 시 백그라운드 데몬이 자동으로 시작됩니다.

| 모드 | 검색 시간 | 메모리 |
|------|-----------|--------|
| 직접 검색 (데몬 없음) | ~3.5초 | - |
| **데몬 (유휴)** | **~2ms** | **~148MB** |
| 데몬 (모델 로드됨) | ~2ms | ~650MB |

- **mmap 스냅샷** — HNSW/BM25 인덱스를 mmap으로 즉시 로딩 (heap 복사 없음)
- **모델 지연 로드** — 첫 쿼리 시 로드, 5분 idle 후 자동 언로드
- **쿼리 캐시** — LRU 1000개, cache hit 시 ~2ms

```bash
# 수동 관리 (선택사항)
v-hnsw serve my-db                  # 데몬 시작
v-hnsw serve my-db --port 9000      # 포트 지정
v-hnsw serve my-db --timeout 600    # idle 타임아웃 (초)
v-hnsw serve my-db --background     # 백그라운드 실행
```

## 지원 입력 포맷

| 포맷 | 확장자 | 설명 |
|------|--------|------|
| **마크다운 폴더** | `*.md` | 시맨틱 청킹, 프론트매터 태그 파싱 |
| JSONL | `.jsonl`, `.ndjson` | `{"text": "...", "tags": [...], "source": "..."}` |
| Parquet | `.parquet` | text/content 컬럼 자동 감지 |

## Low-level 명령어

```bash
# DB 수동 생성 (add는 자동 생성하므로 보통 불필요)
v-hnsw create my-db --dim 256 --metric cosine --neighbors 16 --ef 200

# 벡터 삽입 (텍스트 자동 임베딩)
v-hnsw insert my-db -i data.jsonl --embed

# Raw 벡터 삽입 (사전 임베딩된 데이터, fvecs/bvecs 지원)
v-hnsw insert my-db -i vectors.fvecs

# Raw 벡터 검색
v-hnsw find my-db --vector "0.1,0.2,..." -k 10
```

> `add`는 마크다운 청킹+자동 임베딩+DB 생성+file index 추적을 모두 처리하는 고수준 명령어이고,
> `insert`는 raw 벡터나 커스텀 모델이 필요할 때 사용하는 저수준 명령어입니다.

## 아키텍처

```
crates/
├── v-hnsw-core      # 핵심 트레잇, 타입 (VectorIndex, VectorStore, PayloadStore)
├── v-hnsw-graph     # HNSW 그래프 + mmap 스냅샷 (NodeGraph trait)
├── v-hnsw-search    # 하이브리드 검색 (BM25 + Convex Fusion + MaxScore)
├── v-hnsw-storage   # mmap 벡터 저장소 + WAL + 페이로드 + zstd 텍스트 압축
├── v-hnsw-embed     # model2vec 임베딩
└── v-hnsw-cli       # 통합 CLI (bin: v-hnsw) + 데몬 서버
```

### 검색 파이프라인

```
Query → Embedding (model2vec, 256d)
     ├─ Dense: HNSW search (mmap snapshot, ef=200)
     ├─ Sparse: BM25 search (MaxScore + FST term dict)
     ├─ Dense-Guided BM25 enrichment
     └─ Convex Fusion (alpha * dense + (1-alpha) * sparse)
         → Tag filtering (Roaring Bitmap)
         → Results
```

### 성능 최적화

- **Zero-copy 검색** — mmap 스냅샷에서 bytemuck zero-copy 슬라이스
- **BM25 MaxScore** — non-essential term 스킵으로 대규모 쿼리 가속
- **Dense-Guided BM25** — dense 후보에 대해서만 sparse scoring
- **Zstd 텍스트 압축** — dictionary 기반 ~2.8x 압축
- **Roaring Bitmap** — 태그 인덱스 교집합 연산
- **쿼리 캐시** — LRU 1000개, bincode 직렬화 영속화

## 개발

```bash
# 테스트
cargo test --workspace

# 특정 crate
cargo test -p v-hnsw-graph
cargo test -p v-hnsw-search

# 릴리즈 빌드
cargo build --release -p v-hnsw-cli
```

## 라이선스

MIT OR Apache-2.0
