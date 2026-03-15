# v-hnsw

Rust로 작성된 로컬 벡터 데이터베이스 + 코드 인텔리전스 플랫폼.

12개 크레이트, 259개 파일, 35,000+ 라인. 971개 프로덕션 함수, 1,351개 테스트.

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
# 문서 검색 CLI
cargo build --release -p v-hnsw

# 코드 인텔리전스 CLI
cargo build --release -p v-code

```

빌드된 바이너리: `target/release/v-hnsw`, `target/release/v-code`

> `v-daemon`은 자동으로 시작/관리되므로 별도 빌드 불필요. 필요 시 `cargo build --release -p v-daemon`.

### 첫 실행 시 자동 다운로드

첫 `add` 또는 `find` 실행 시 아래 항목이 자동 다운로드됩니다:

- **임베딩 모델** — `minishlab/potion-multilingual-128M` (~500MB, HuggingFace Hub)
- **한국어 사전** — `ko-dic` (Lindera, `~/.v-hnsw/dict/ko-dic/`)

이후에는 로컬 캐시를 사용합니다.

## 주요 기능

### 하이브리드 검색 엔진

- **HNSW + BM25 + Convex Fusion** — 벡터 유사도와 키워드 매칭 동시 활용
- **자동 임베딩** — model2vec 기반 (100+ 언어, 256차원)
- **SQ8 양자화** — f32→u8 4x 메모리 절감, 2단계 검색 (SQ8 탐색 → f32 rescore)
- **한국어 지원** — Lindera ko-dic 형태소 분석
- **태그 필터링** — 마크다운 프론트매터 태그 기반 Roaring Bitmap 필터

### 코드 인텔리전스

- **tree-sitter 기반 심볼 추출** — 10개 언어 (Rust, TS, JS, Python, Go, Java, C, C++, C#, Ruby)
- **호출 그래프** — callee/caller BFS, import 기반 모듈 해석, trait impl 연결
- **영향 분석 (blast radius)** — 심볼 변경 시 전파 범위 + prod/test 분리
- **클론 감지** — AST 해시 + MinHash Jaccard + sub-block 비교
- **문자열 흐름 추적** — 함수 간 string literal 전파 경로 분석

### 데몬 + 파일 감시

- **mmap 스냅샷** — 즉시 로딩, ~2ms 검색
- **파일 감시 (notify)** — 소스 파일 변경 자동 감지, graph 캐시 즉시 무효화
- **자동 재인덱싱** — 파일 변경 시 `v-code add` 백그라운드 실행 (debounce 2초)
- **증분 업데이트** — 변경된 파일만 감지하여 재인덱싱

## 사용법

```bash
# 1. 데이터 추가 (마크다운 폴더)
v-hnsw add my-db ./documents/

# 2. 검색
v-hnsw find my-db "검색하고 싶은 내용"

# 3. 변경된 파일만 업데이트
v-hnsw update my-db
```

모델 다운로드, DB 생성, 인덱싱 모두 자동으로 처리됩니다.

### v-hnsw 서브커맨드 (문서 검색)

| 커맨드 | 설명 |
|--------|------|
| `add` | 마크다운/JSONL/Parquet/소스코드 추가 (청킹+임베딩+인덱싱) |
| `find` | 하이브리드 검색 (데몬 자동 시작) |
| `update` | 변경 파일만 증분 업데이트 |
| `info` | DB 정보 (문서 수, 크기, 설정) |
| `get` | ID로 문서 조회 |
| `delete` | 문서 삭제 |
| `build-index` | 인덱스 재빌드 (HNSW + BM25 + SQ8 + 스냅샷) |
| `bench` | 벤치마크 (brute-force ground truth + recall@k + QPS) |
| `export` | JSONL 내보내기 |
| `collection` | 컬렉션 관리 |

### v-code 서브커맨드 (코드 인텔리전스)

| 커맨드 | 별칭 | 설명 |
|--------|------|------|
| `add` | | 코드 파일 인덱싱 (tree-sitter 청킹 + 임베딩) |
| `context` | `ctx` | 통합 컨텍스트 (정의 + callers + callees + types + tests) |
| `blast` | `bl` | 영향 분석 (transitive callers + prod/test 분리) |
| `jump` | `j` | 실행 흐름 트리 (DFS callee 탐색) |
| `trace` | `tr` | 두 심볼 간 최단 호출 경로 |
| `symbols` | | 심볼 목록 (함수, 구조체, enum 등) |
| `deps` | | 파일 의존성 그래프 |
| `detail` | `dt` | 설계 판단/이력 조회·관리 |
| `dupes` | `dup` | 중복 코드 감지 (AST hash + MinHash + sub-block) |
| `strings` | `str` | 문자열 리터럴 검색 |
| `flow` | `fl` | 함수 간 문자열 흐름 추적 |
| `stats` | | 크레이트별 코드 통계 |

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

# 최소 점수 필터
v-hnsw find my-db "검색어" --min-score 0.5
```

### add — 데이터 추가

```bash
# 마크다운 폴더
v-hnsw add my-db ./notes/

# 소스코드 폴더 (tree-sitter 기반 청킹)
v-hnsw add my-code.db ./src/

# JSONL / Parquet 파일
v-hnsw add my-db data.jsonl

# 특정 디렉토리 제외
v-hnsw add my-db ./project/ --exclude node_modules --exclude .git
```

입력 타입 자동 감지, 시맨틱 청킹, 임베딩, 인덱싱까지 한 번에 처리합니다.

### v-code — 코드 인텔리전스

```bash
# 코드 인덱싱
v-code add my-code.db ./src/ --exclude target

# 통합 컨텍스트 (정의 + callers + callees + types + tests)
v-code context my-code.db search_layer
v-code context my-code.db search_layer --depth 2

# 영향 분석 (변경 시 전파 범위)
v-code blast my-code.db HnswGraph --depth 3
v-code blast my-code.db HnswGraph --include-tests

# 실행 흐름 트리
v-code jump my-code.db main --depth 3

# 두 심볼 간 호출 경로
v-code trace my-code.db main search_layer

# 심볼 검색
v-code symbols my-code.db --kind function
v-code symbols my-code.db -n search

# 중복 코드 감지 + 분석
v-code dupes my-code.db --all --exclude-tests
v-code dupes my-code.db --all --analyze    # callee/blast 교차 분석 포함

# 문자열 흐름 추적
v-code strings my-code.db "error"
v-code flow my-code.db "connection refused"

# 설계 판단 기록
v-code detail my-code.db HnswGraph --decision "mmap snapshot for zero-copy"
```

### bench — 벤치마크

```bash
# recall@k + QPS 측정 (f32 vs SQ8)
v-hnsw bench my-db -q 100 -k 10
```

출력 예시:
```
Database: 2482 vectors, dim=256
Queries:  100 sampled vectors

=== HNSW f32 ===
  ef= 50: recall@10=0.9740    7839 QPS    124us/q
  ef=200: recall@10=0.9960    2956 QPS    338us/q

=== HNSW SQ8 (two-stage) ===
  Memory: 0.61MB (SQ8) vs 2.42MB (f32) = 4.0x compression
  ef= 50: recall@10=0.9740    5359 QPS    187us/q
  ef=200: recall@10=0.9960    1871 QPS    535us/q
```

## 검색 결과 예시

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

`find` 실행 시 백그라운드 데몬이 **자동으로** 시작됩니다. 사용자가 직접 관리할 필요 없습니다.

| 모드 | 검색 시간 | 메모리 |
|------|-----------|--------|
| 직접 검색 (데몬 없음) | ~3.5초 | - |
| **데몬 (유휴)** | **~2ms** | **~148MB** |
| 데몬 (모델 로드됨) | ~2ms | ~650MB |

- **mmap 스냅샷** — HNSW/BM25/SQ8 인덱스를 mmap으로 즉시 로딩 (heap 복사 없음)
- **모델 지연 로드** — 첫 쿼리 시 로드, 5분 idle 후 자동 언로드
- **쿼리 캐시** — LRU 1000개, cache hit 시 ~2ms
- **SQ8 자동 감지** — sq8 파일 존재 시 2단계 검색 활성화, 없으면 f32 fallback
- **파일 감시** — `notify` 기반 소스 파일 변경 감지, graph 캐시 즉시 무효화
- **자동 재인덱싱** — 변경 감지 2초 debounce 후 `v-code add` 백그라운드 실행
- **바이너리 갱신 감지** — `cargo install` 후 데몬 자동 재시작

## 지원 입력 포맷

| 포맷 | 확장자 | 설명 |
|------|--------|------|
| **마크다운 폴더** | `*.md` | 시맨틱 청킹, 프론트매터 태그 파싱 |
| **소스코드 폴더** | `*.rs`, `*.ts`, `*.py` 등 | tree-sitter 함수/구조체 단위 청킹 (10개 언어) |
| JSONL | `.jsonl`, `.ndjson` | `{"text": "...", "tags": [...], "source": "..."}` |
| Parquet | `.parquet` | text/content 컬럼 자동 감지 |

### 지원 언어 (소스코드)

Rust, TypeScript, JavaScript, Python, Go, Java, C, C++, C#, Ruby

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

> `add`는 마크다운/소스코드 청킹+자동 임베딩+DB 생성+file index 추적을 모두 처리하는 고수준 명령어이고,
> `insert`는 raw 벡터나 커스텀 모델이 필요할 때 사용하는 저수준 명령어입니다.

## 아키텍처

```
crates/
├── v-hnsw-core      # 핵심 트레잇, 타입 (VectorIndex, VectorStore, PayloadStore)
├── v-hnsw-graph     # HNSW 그래프 + mmap 스냅샷 + DistanceComputer trait
├── v-hnsw-search    # 하이브리드 검색 (BM25 + Convex Fusion + MaxScore)
├── v-hnsw-storage   # mmap 벡터 저장소 + WAL + SQ8 양자화 + 데몬 클라이언트
├── v-hnsw-embed     # model2vec 임베딩
├── v-hnsw-rerank    # cross-encoder reranking (ms-marco-TinyBERT)
├── v-hnsw-cli       # 공유 인프라 (lib) + v-hnsw 문서 검색 바이너리
├── v-code-chunk     # tree-sitter 코드 청킹 (10개 언어)
├── v-code-intel     # 호출 그래프, 영향 분석, 클론 감지, 문자열 흐름
├── v-code           # v-code 코드 인텔리전스 바이너리
└── v-daemon         # 백그라운드 데몬 (mmap 캐시 + 파일 감시 + 자동 재인덱싱)
```

### 검색 파이프라인

```
Query → Embedding (model2vec, 256d)
     ├─ Dense: HNSW search (SQ8 two-stage or f32, mmap snapshot)
     ├─ Sparse: BM25 search (MaxScore + FST term dict)
     ├─ Dense-Guided BM25 enrichment
     └─ Convex Fusion (alpha * dense + (1-alpha) * sparse)
         → Tag filtering (Roaring Bitmap)
         → Results
```

### SQ8 양자화

```
Build:  f32 vectors → train min/max → quantize → sq8_vectors.bin (4x smaller)
Search: SQ8 asymmetric distance (graph traversal) → f32 exact distance (rescore)
```

- 차원별 [min, max] → [0, 255] 선형 매핑
- LUT 기반 비대칭 거리: query는 f32, DB 벡터는 u8
- recall 손실 없음 (2단계 rescore), 메모리 4x 절감

### 성능 최적화

- **SQ8 양자화** — f32→u8 4x 메모리 절감, LUT 비대칭 거리
- **Zero-copy 검색** — mmap 스냅샷에서 bytemuck zero-copy 슬라이스
- **BM25 MaxScore** — non-essential term 스킵으로 대규모 쿼리 가속
- **Dense-Guided BM25** — dense 후보에 대해서만 sparse scoring
- **FSST + Zstd 텍스트 압축** — dictionary 기반 ~5x 압축
- **Roaring Bitmap** — 태그 인덱스 교집합 연산
- **쿼리 캐시** — LRU 1000개, bincode 직렬화 영속화
- **Software Prefetch** — HNSW 탐색 시 다음 이웃 벡터 L1 프리페치

## 개발

```bash
# 테스트 (nextest 사용)
cargo nextest run --workspace

# 특정 crate
cargo nextest run -p v-hnsw-graph

# 특정 테스트
cargo nextest run -E "test(search_two_stage)"

# 벤치마크
v-hnsw bench my-db -q 100 -k 10

# 릴리즈 빌드
cargo build --release -p v-hnsw -p v-code
```

## 라이선스

MIT OR Apache-2.0
