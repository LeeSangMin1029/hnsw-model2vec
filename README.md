# v-hnsw

Rust로 작성된 로컬 벡터 데이터베이스. HNSW 알고리즘 기반 유사도 검색과 BM25 키워드 검색을 결합한 하이브리드 검색을 지원합니다.

## 주요 특징

- **올인원 CLI** — 데이터 추가, 검색, 증분 업데이트를 간단한 명령어로
- **자동 임베딩** — model2vec 기반 텍스트 임베딩 (100+ 언어, 256차원)
- **하이브리드 검색** — HNSW 벡터 + BM25 키워드 + Convex Fusion
- **데몬 모드** — mmap 스냅샷으로 즉시 로딩, 모델 지연 로드
- **한국어 지원** — Lindera ko-dic 형태소 분석 BM25 토크나이저
- **증분 업데이트** — 변경된 파일만 감지하여 재인덱싱

---

## 빠른 시작

```bash
# 1. 데이터 추가 (마크다운 폴더)
v-hnsw add my-db ./documents/

# 2. 검색
v-hnsw find my-db "검색하고 싶은 내용"

# 3. 변경된 파일만 업데이트
v-hnsw update my-db ./documents/
```

모델 다운로드, DB 생성, 인덱싱 모두 자동으로 처리됩니다.

---

## 설치

### 요구 사항

- Rust 1.92+ (Edition 2024)
- 한국어 사전은 첫 실행 시 자동 다운로드 (`~/.v-hnsw/dict/ko-dic/`)

### 소스에서 빌드

```bash
cargo build --release -p v-hnsw-cli
```

빌드된 바이너리: `target/release/v-hnsw` (Linux/macOS) / `target/release/v-hnsw.exe` (Windows)

PATH에 추가:

```bash
# Linux / macOS
cp target/release/v-hnsw ~/.local/bin/

# Windows (PowerShell)
Copy-Item target\release\v-hnsw.exe "$env:USERPROFILE\.cargo\bin\"
```

### 첫 실행

첫 `add` 또는 `find` 실행 시:
1. **한국어 사전** — `~/.v-hnsw/dict/ko-dic/`에 자동 다운로드+빌드
2. **임베딩 모델** — `minishlab/potion-multilingual-128M` (~500MB, HuggingFace Hub에서 자동 다운로드)

이후에는 로컬 캐시를 사용합니다.

---

## 사용법

### 데이터 추가 (`add`)

```bash
# 마크다운 폴더
v-hnsw add my-db ./notes/

# JSONL 파일
v-hnsw add my-db data.jsonl

# Parquet 파일
v-hnsw add my-db data.parquet
```

입력 타입 자동 감지, 시맨틱 청킹, 임베딩, 인덱싱까지 한 번에 처리합니다.

### 검색 (`find`)

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

### 증분 업데이트 (`update`)

```bash
v-hnsw update my-db ./documents/
```

파일 수정 시간/크기를 비교하여 새로운/변경된/삭제된 파일만 처리합니다.

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

---

## 데몬 모드

`find` 실행 시 백그라운드 데몬이 자동으로 시작됩니다.

| 모드 | 검색 시간 | 메모리 |
|------|-----------|--------|
| 직접 검색 (데몬 없음) | ~3.5초 | - |
| **데몬 (유휴)** | **~2ms** | **~148MB** |
| 데몬 (모델 로드됨) | ~2ms | ~650MB |

- **mmap 스냅샷** — HNSW/BM25 인덱스를 mmap으로 즉시 로딩 (heap 복사 없음)
- **모델 지연 로드** — 첫 쿼리 시 로드, 5분 idle 후 자동 언로드
- **쿼리 캐시** — LRU 1000개, 임베딩 추론 스킵으로 cache hit 시 ~2ms

```bash
# 수동 관리 (선택사항)
v-hnsw serve my-db                  # 데몬 시작
v-hnsw serve my-db --port 9000      # 포트 지정
v-hnsw serve my-db --timeout 600    # idle 타임아웃 (초)
v-hnsw serve my-db --background     # 백그라운드 실행
```

데몬은 JSON-RPC over TCP로 통신하며, `search`, `embed`, `ping`, `reload`, `shutdown` 메서드를 지원합니다.

---

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

---

## 기타 명령어

```bash
v-hnsw info my-db                    # DB 정보
v-hnsw get my-db 42 43 44            # ID로 문서 조회
v-hnsw delete my-db --id 42          # 문서 삭제
v-hnsw build-index my-db             # 인덱스 재빌드 (스냅샷 포함)
v-hnsw export my-db -o backup.jsonl  # JSONL 내보내기
v-hnsw import my-db -i backup.jsonl  # JSONL 가져오기
```

### Low-level 명령어

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

> `add`는 마크다운 청킹·자동 임베딩·DB 생성·file index 추적을 모두 처리하는 고수준 명령어이고,
> `insert`는 raw 벡터나 커스텀 모델이 필요할 때 사용하는 저수준 명령어입니다.

---

## 지원 입력 포맷

| 포맷 | 확장자 | 설명 |
|------|--------|------|
| **마크다운 폴더** | `*.md` | 시맨틱 청킹, 프론트매터 태그 파싱 |
| JSONL | `.jsonl`, `.ndjson` | `{"text": "...", "tags": [...], "source": "..."}` |
| Parquet | `.parquet` | text/content 컬럼 자동 감지 |

---

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
- **Zstd 텍스트 압축** — dictionary 기반 2.8x 압축 (51MB → 18MB)
- **Roaring Bitmap** — 태그 인덱스 SIMD 교집합 연산
- **쿼리 캐시** — LRU 1000개, bincode 직렬화 영속화

---

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

---

## 라이선스

MIT OR Apache-2.0
