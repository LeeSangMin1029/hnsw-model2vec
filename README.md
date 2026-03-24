# v-hnsw

Rust로 작성된 로컬 벡터 데이터베이스 + 코드 인텔리전스 플랫폼.

10개 크레이트, 240개 파일, 40,000+ 라인 (프로덕션 24K + 테스트 17K). 1,548개 프로덕션 함수, 1,128개 테스트.

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

### v-code 한 줄 설치

```bash
curl -sL https://raw.githubusercontent.com/LeeSangMin1029/hnsw-model2vec/main/install-vcode.sh | bash
```

바이너리 다운로드 + nightly Rust 설치를 자동 처리합니다. clone/빌드 불필요.

설치 후 아무 Rust 프로젝트에서:
```bash
cd /path/to/your/project
v-code add .code.db .
v-code context .code.db YourFunction -s
v-code context .code.db YourFunction --blast
```

> `mir-callgraph` 바이너리는 첫 `v-code add` 실행 시 **자동 빌드**됩니다 (~10초, 1회).

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

- **MIR 기반 호출 그래프** — rustc MIR에서 call edges 추출 (100% 정확, trait dispatch/generic/closure 해소)
- **Direct 모드** — rustc args 캐싱 + cargo 우회로 증분 업데이트 ~1초 (44K LOC), ~4.5초 (549K LOC)
- **비동기 test** — lib만 대기 후 즉시 반환, test는 백그라운드 실행 (100% 정확도 유지)
- **영향 분석 (blast radius)** — 심볼 변경 시 전파 범위 + prod/test 분리
- **클론 감지** — AST 해시 + MinHash Jaccard + sub-block 비교
- **dead code 탐지** — caller 없는 함수 자동 검출
- **테스트 커버리지** — `cargo llvm-cov` 기반 실제 커버리지 + 정적 reachability fallback
- **심볼 기반 편집** — replace, insert, delete를 심볼 단위로 수행 (file lock으로 동시성 안전)

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

#### 분석

| 커맨드 | 별칭 | 설명 |
|--------|------|------|
| `add` | | MIR 기반 코드 인덱싱 (nightly 필요, 증분 지원) |
| `context` | `ctx` | 통합 컨텍스트 (정의 + callers + callees + types + tests, `--blast`로 영향 분석) |
| `trace` | `tr` | 두 심볼 간 최단 호출 경로 |
| `dupes` | `dup` | 중복 코드 감지 (AST hash + MinHash + sub-block) |
| `dead` | | caller 없는 함수 탐지 (unreachable code) |
| `coverage` | `cov` | 테스트 커버리지 (llvm-cov 기반) |
| `symbols` | | 심볼 목록 (함수, 구조체, enum 등) |
| `stats` | | 크레이트별 코드 통계 |
| `aliases` | | 경로 별칭 매핑 |
| `watch` | | 파일 변경 자동 감시 + 증분 업데이트 |
| `embed` | | 시맨틱 검색용 벡터 임베딩 |

#### 편집 (심볼 기반, 동시성 안전)

| 커맨드 | 별칭 | 설명 |
|--------|------|------|
| `replace` | `rep` | 심볼 본체 교체 |
| `insert-after` | | 심볼 뒤에 삽입 |
| `insert-before` | | 심볼 앞에 삽입 |
| `delete-symbol` | `del` | 심볼 삭제 |
| `insert-at` | `ia` | 특정 라인에 삽입 |
| `replace-lines` | `rl` | 라인 범위 교체 |
| `delete-lines` | `dl` | 라인 범위 삭제 |
| `create-file` | `cf` | 새 파일 생성 |

모든 편집 커맨드는 `.lock` sidecar file 기반 exclusive lock으로 동시 편집 안전.

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

# 소스코드 폴더
v-hnsw add my-code.db ./src/

# JSONL / Parquet 파일
v-hnsw add my-db data.jsonl

# 특정 디렉토리 제외
v-hnsw add my-db ./project/ --exclude node_modules --exclude .git
```

입력 타입 자동 감지, 시맨틱 청킹, 임베딩, 인덱싱까지 한 번에 처리합니다.

### v-code — 코드 인텔리전스

```bash
# 코드 인덱싱 (nightly rustc 필요)
v-code add .code.db .

# 통합 컨텍스트 (정의 + callers + callees + types + tests)
v-code context .code.db search_layer
v-code context .code.db search_layer -s          # 소스 코드 인라인
v-code context .code.db search_layer --tree       # DFS callee 트리

# 영향 분석 (변경 시 전파 범위)
v-code context .code.db HnswGraph --blast --depth 3
v-code context .code.db HnswGraph --blast --include-tests

# 두 심볼 간 호출 경로
v-code trace .code.db main search_layer

# dead code 탐지
v-code dead .code.db

# 테스트 커버리지
v-code coverage .code.db

# 중복 코드 감지 + 분석
v-code dupes .code.db --all --analyze

# 심볼 기반 편집
v-code replace .code.db my_function --body-file /tmp/new_body.rs
v-code insert-after .code.db my_function --body-file /tmp/code.rs
v-code delete-symbol .code.db unused_function

# 파일 변경 자동 감시
v-code watch .code.db .
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

## 동시성

여러 에이전트(프로세스)가 v-code를 동시에 사용할 수 있습니다:

- **읽기 커맨드** (context, blast, trace, symbols 등) — 동시 실행 안전
- **편집 커맨드** (replace, insert-*, delete-*) — `.lock` sidecar file 기반 exclusive lock으로 같은 파일 동시 편집 안전
- **DB 쓰기** (add) — 단일 프로세스 권장

10개 에이전트 동시 편집 × 50 라운드 스트레스 테스트 통과.

## 지원 입력 포맷

| 포맷 | 확장자 | 설명 |
|------|--------|------|
| **마크다운 폴더** | `*.md` | 시맨틱 청킹, 프론트매터 태그 파싱 |
| **소스코드 폴더** | `*.rs` 등 | MIR 기반 함수/구조체 단위 인덱싱 |
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
├── v-hnsw-cli       # CLI 공유 인프라 (lib) + 문서 검색 커맨드
├── v-hnsw           # v-hnsw 문서 검색 바이너리 (thin wrapper)
├── v-code-intel     # 호출 그래프, 영향 분석, 클론 감지, 동시성 테스트
├── v-code           # v-code 코드 인텔리전스 바이너리 + 심볼 기반 편집
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

### 코드 인텔리전스 파이프라인

```
Source → cargo +nightly check (RUSTC_WRAPPER=mir-callgraph)
     → MIR call edges + chunk 정보 추출
     → CallGraph (callee/caller/trait_impl 양방향 인접 리스트)
     → context / blast / trace / dead / coverage / dupes
```

증분 업데이트 (direct 모드):
```
파일 변경 → detect_changed_crates → rustc_driver 직접 호출 (cargo 우회)
         → lib 동기 완료 → graph 즉시 업데이트 → 사용 가능
         → test 백그라운드 완료 → graph 보강 (자동)
```

#### 성능 (실측)

| 프로젝트 | LOC | Cold start | 증분 | No-op |
|---------|-----|-----------|------|-------|
| v-code | 44K | 44s | **0.95s** | 0.12s |
| rust-analyzer | 549K | 77s | **4.5s** | 0.22s |

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
