# v-hnsw

Rust로 작성된 로컬 벡터 데이터베이스. HNSW 알고리즘 기반의 빠른 유사도 검색과 BM25 키워드 검색을 결합한 하이브리드 검색을 지원합니다.

## 주요 특징

- **올인원 CLI** - 데이터 추가, 검색, 증분 업데이트를 간단한 명령어로
- **자동 임베딩** - model2vec 기반 텍스트 임베딩 (100+ 언어 지원, 1024차원)
- **하이브리드 검색** - HNSW 벡터 검색 + BM25 키워드 검색 + RRF 융합
- **데몬 모드** - 모델/인덱스를 메모리에 유지, 데이터 변경 시 자동 리로드
- **증분 업데이트** - 변경된 파일만 감지하여 재인덱싱
- **태그 필터링** - 마크다운 프론트매터 태그 기반 검색 필터
- **한국어 지원** - Lindera ko-dic 기반 형태소 분석 BM25

---

## 빠른 시작

```bash
# 1. 데이터 추가 (폴더 또는 파일)
v-hnsw add my-db ./documents/

# 2. 검색
v-hnsw find my-db "검색하고 싶은 내용"

# 3. 변경된 파일만 업데이트
v-hnsw update my-db ./documents/
```

모델 다운로드, DB 생성, 인덱싱 모두 자동으로 처리됩니다.

---

## 설치

### 소스에서 빌드

```bash
# 요구 사항: Rust 1.92+ (Edition 2024)

# CPU 버전
cargo build --release -p v-hnsw-cli

# 빌드된 바이너리
# target/release/v-hnsw (Linux/macOS)
# target/release/v-hnsw.exe (Windows)
```

빌드 후 바이너리를 PATH에 추가하면 어디서든 사용할 수 있습니다.

**Windows (PowerShell):**
```powershell
# 예: 사용자 로컬 bin 디렉토리에 복사
$binDir = "$env:USERPROFILE\.local\bin"
New-Item -ItemType Directory -Force -Path $binDir | Out-Null
Copy-Item target\release\v-hnsw.exe $binDir\

# PATH에 추가 (영구)
[Environment]::SetEnvironmentVariable("Path", "$env:Path;$binDir", "User")
```

**Linux / macOS:**
```bash
# 예: ~/.local/bin에 복사
mkdir -p ~/.local/bin
cp target/release/v-hnsw ~/.local/bin/

# PATH에 추가 (~/.bashrc 또는 ~/.zshrc)
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
```

### 첫 실행 시 모델 다운로드

첫 `add` 또는 `find` 실행 시 model2vec 모델(`minishlab/potion-multilingual-128M`, ~500MB)이 HuggingFace Hub에서 자동 다운로드됩니다. 이후에는 로컬 캐시를 사용합니다.

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
- DB 생성 (없으면 자동 생성)
- 입력 타입 감지 (폴더/JSONL/Parquet)
- 마크다운 시맨틱 청킹 (프론트매터 태그 파싱 포함)
- model2vec 임베딩 (1024차원, 100+ 언어)
- HNSW + BM25 인덱스 빌드
- 데몬 실행 중이면 자동 리로드 알림

### 검색 (`find`)

```bash
# 기본 검색 (하이브리드: HNSW 70% + BM25 30%)
v-hnsw find my-db "API 보안 모범 사례"

# 결과 수 조절
v-hnsw find my-db "검색어" -k 20

# 태그 필터링
v-hnsw find my-db "프로젝트 구조" --tag project-structure

# 복수 태그 (AND 조건)
v-hnsw find my-db "검색어" --tag rust --tag architecture
```

**검색 방식**: 하이브리드 (HNSW 벡터 70% + BM25 키워드 30% + RRF 융합)

### 증분 업데이트 (`update`)

```bash
# 변경된 파일만 재인덱싱
v-hnsw update my-db ./documents/
```

파일의 수정 시간과 크기를 비교하여 새로운/변경된/삭제된 파일만 처리합니다. 해당 입력 폴더 범위 내의 파일만 삭제 대상으로 판단합니다.

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
  "model": "(daemon)",
  "total_docs": 2768,
  "elapsed_ms": 1.2
}
```

---

## 데몬 모드

`find` 실행 시 백그라운드 데몬이 자동으로 시작됩니다. 데몬은 모델과 인덱스를 메모리에 유지하여 검색 속도를 크게 향상시킵니다.

| 상태 | 검색 시간 |
|------|-----------|
| 데몬 없음 (직접 검색) | ~3.5초 |
| **데몬 사용** | **~1ms** |

**자동 동작:**
- 첫 `find`: 데몬 자동 시작 + 모델/인덱스 로딩
- 이후 `find`: 즉시 응답 (~1ms)
- `add`/`update` 후: 데몬에 자동 리로드 알림 (모델 재로드 없이 인덱스만 갱신)
- 5분 idle: 자동 종료 (메모리 반환)

```bash
# 수동 데몬 관리 (선택사항)
v-hnsw serve my-db              # 데몬 시작
v-hnsw serve my-db --port 9000  # 특정 포트로 시작
v-hnsw serve my-db --timeout 600 # idle 타임아웃 조절 (초)
v-hnsw serve my-db --background # 백그라운드 실행
```

데몬은 JSON-RPC 프로토콜(TCP)로 통신하며, `search`, `ping`, `reload`, `shutdown` 메서드를 지원합니다.

---

## 태그 시스템

마크다운 파일의 YAML 프론트매터에 태그를 지정하면 검색 시 필터링할 수 있습니다.

```markdown
---
tags: [project-structure, rust, architecture]
title: 프로젝트 아키텍처
---

# 프로젝트 구조
...
```

```bash
# 태그로 필터링된 검색
v-hnsw find my-db "아키텍처" --tag project-structure
```

태그 필터링 시 내부적으로 후보를 10배 많이 가져온 후 필터링하여 정확도를 보장합니다.

---

## 고급 사용법

### 세부 옵션이 필요한 경우

```bash
# 데이터베이스 직접 생성
v-hnsw create my-db --dim 1024 --metric cosine --neighbors 16 --ef 200

# 상세 옵션으로 삽입
v-hnsw insert my-db -i data.jsonl --embed --model minishlab/potion-multilingual-128M --batch-size 1024

# 벡터 직접 검색
v-hnsw search my-db --vector "0.1,0.2,..." --k 10 --ef 500

# 텍스트로 검색 (BM25만)
v-hnsw search my-db --text "키워드 검색" --k 10

# 시맨틱 벡터 검색 (HNSW만)
v-hnsw vsearch my-db "검색 쿼리" --show-text
```

### 기타 명령어

```bash
v-hnsw info my-db                    # DB 정보 (벡터 수, 차원, 설정)
v-hnsw get my-db 42 43 44            # ID로 문서 조회
v-hnsw delete my-db --id 42          # 문서 삭제
v-hnsw build-index my-db             # 인덱스 재빌드
v-hnsw export my-db -o backup.jsonl  # JSONL로 내보내기
v-hnsw import my-db -i backup.jsonl  # JSONL에서 가져오기
```

### 컬렉션 관리

```bash
v-hnsw collection my-db create my-collection --dim 1024
v-hnsw collection my-db list
v-hnsw collection my-db delete my-collection
```

---

## 지원 임베딩 모델

기본 모델은 `minishlab/potion-multilingual-128M`이며, model2vec 기반으로 HuggingFace Hub의 model2vec 모델을 사용할 수 있습니다.

| 모델 | 차원 | 언어 | 비고 |
|------|------|------|------|
| **minishlab/potion-multilingual-128M** (기본) | 1024 | 100+ | 한국어 포함, model2vec |
| minishlab/potion-base-8M | 256 | 영어 | 경량 |

model2vec는 정적 임베딩 모델로, GPU 없이도 빠르게 동작합니다.

---

## 지원 입력 포맷

| 포맷 | 확장자 | 설명 |
|------|--------|------|
| **마크다운 폴더** | `*.md` | 자동 시맨틱 청킹, 프론트매터 태그 파싱 |
| JSONL | `.jsonl`, `.ndjson` | `{"text": "...", "tags": [...], "source": "..."}` |
| Parquet | `.parquet` | text/content 컬럼 자동 감지 |

### JSONL 형식

```json
{"text": "문서 내용", "source": "file.md", "title": "제목", "tags": ["tag1", "tag2"]}
```

### 마크다운 청킹 설정

- 타겟 청크 크기: 1000자
- 오버랩: 200자
- 최소 크기: 100자
- 헤딩 컨텍스트 포함

---

## 개발

### 테스트

```bash
# 전체 워크스페이스 테스트
cargo test --workspace

# 특정 크레이트만 테스트
cargo test -p v-hnsw-graph
cargo test -p v-hnsw-search
cargo test -p v-hnsw-storage

# 특정 테스트 함수만 실행
cargo test -p v-hnsw-graph -- test_name

# 벤치마크 (v-hnsw-graph)
cargo bench -p v-hnsw-graph
```

### 빌드

```bash
# 개발 빌드
cargo build -p v-hnsw-cli

# 릴리즈 빌드
cargo build --release -p v-hnsw-cli

# CUDA GPU 가속 빌드
cargo build --release -p v-hnsw-cli --features cuda
```

---

## 프로젝트 구조

```
crates/
├── v-hnsw-core        # 핵심 트레잇, 타입 (VectorIndex, PayloadStore)
├── v-hnsw-distance    # SIMD 거리 함수 (L2, Cosine, Dot)
├── v-hnsw-graph       # HNSW 그래프 구현 (빌드, 검색, 직렬화)
├── v-hnsw-storage     # mmap 벡터 저장소 + WAL + 컬렉션
├── v-hnsw-search      # 하이브리드 검색 (HNSW + BM25 + RRF 융합)
├── v-hnsw-embed       # 임베딩 모델 (model2vec)
├── v-hnsw-chunk       # 마크다운 시맨틱 청킹
├── v-hnsw-tokenizer   # 한국어 형태소 분석 (Lindera ko-dic)
├── v-hnsw-quantize    # 벡터 양자화
├── v-hnsw-gpu         # GPU 가속 (cubecl)
├── v-hnsw             # 라이브러리 크레이트 (통합 API)
└── v-hnsw-cli         # CLI 바이너리
```

---

## 한국어 지원

- **BM25 토크나이저**: Lindera ko-dic 기반 형태소 분석
  - "프로젝트 구조 분석" → ["프로젝트", "구조", "분석"]
  - 조사/어미 제거로 정확한 키워드 매칭
- **다국어 임베딩**: model2vec potion-multilingual-128M (100+ 언어)
- 데이터베이스 생성 시 자동으로 한국어 토크나이저 활성화

---

## 라이선스

MIT OR Apache-2.0
