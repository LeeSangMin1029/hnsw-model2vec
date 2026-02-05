# v-hnsw 빠른 시작 가이드

마크다운 파일을 인덱싱하고 시맨틱 검색하는 방법을 안내합니다.

## 1. 설치

### Windows (PowerShell)
```powershell
irm https://github.com/LeeSangMin1029/v-hnsw/releases/latest/download/install.ps1 | iex
```

### Linux/macOS
```bash
curl -fsSL https://github.com/LeeSangMin1029/v-hnsw/releases/latest/download/install.sh | bash
```

### 수동 설치
[GitHub Releases](https://github.com/LeeSangMin1029/v-hnsw/releases)에서 OS에 맞는 바이너리 다운로드 후 PATH에 추가.

---

## 2. 마크다운 파일 인덱싱

### Step 1: 마크다운을 JSONL로 변환

먼저 마크다운 파일들을 JSONL 형식으로 변환합니다.

**스크립트 예시 (PowerShell):**
```powershell
# md-to-jsonl.ps1
$inputDir = "C:\path\to\markdown"
$outputFile = "documents.jsonl"

Get-ChildItem -Path $inputDir -Filter "*.md" -Recurse | ForEach-Object {
    $content = Get-Content $_.FullName -Raw -Encoding UTF8
    $obj = @{
        text = $content
        payload = @{
            path = $_.FullName
            title = $_.BaseName
        }
    }
    $obj | ConvertTo-Json -Compress -Depth 3
} | Out-File -FilePath $outputFile -Encoding UTF8
```

**스크립트 예시 (Bash):**
```bash
#!/bin/bash
# md-to-jsonl.sh
INPUT_DIR="/path/to/markdown"
OUTPUT_FILE="documents.jsonl"

find "$INPUT_DIR" -name "*.md" -type f | while read -r file; do
    content=$(cat "$file" | jq -Rs .)
    path=$(echo "$file" | jq -Rs .)
    title=$(basename "$file" .md | jq -Rs .)
    echo "{\"text\":$content,\"payload\":{\"path\":$path,\"title\":$title}}"
done > "$OUTPUT_FILE"
```

### Step 2: 데이터베이스 생성 및 인덱싱

```bash
# 데이터베이스 생성 (자동으로 생성되지만 명시적으로 할 수도 있음)
v-hnsw create my-docs -d 768 --metric cosine

# JSONL 파일 인덱싱 (자동 임베딩)
v-hnsw insert my-docs -i documents.jsonl --embed --model bge-base-en-v1.5

# GPU 사용 시 (CUDA)
v-hnsw insert my-docs -i documents.jsonl --embed --model bge-base-en-v1.5 --device cuda

# FP16으로 VRAM 절약 (RTX 3060 6GB 권장)
v-hnsw insert my-docs -i documents.jsonl --embed --model bge-base-en-v1.5 --device cuda --fp16
```

### Step 3: HNSW 인덱스 빌드

```bash
v-hnsw build-index my-docs
```

---

## 3. 검색

### 시맨틱 검색 (자연어 쿼리)

```bash
# 기본 검색
v-hnsw vsearch my-docs "검색하고 싶은 내용"

# 결과 수 조절
v-hnsw vsearch my-docs "검색 쿼리" -k 20

# 텍스트 내용 표시
v-hnsw vsearch my-docs "검색 쿼리" --show-text
```

### 출력 예시

```json
{
  "results": [
    {"id": 42, "score": 0.89, "text": "관련 문서 내용..."},
    {"id": 17, "score": 0.85, "text": "다른 관련 내용..."}
  ],
  "query": "검색 쿼리",
  "model": "bge-base-en-v1.5",
  "elapsed_ms": 45.2
}
```

### 하이브리드 검색 (벡터 + 키워드)

```bash
# 벡터 검색
v-hnsw search my-docs -v "0.1,0.2,..." -k 10

# 키워드 검색
v-hnsw search my-docs -t "키워드" -k 10

# 하이브리드 + 리랭킹
v-hnsw search my-docs -v "..." -t "키워드" --rerank
```

---

## 4. 데이터베이스 관리

### 정보 확인
```bash
v-hnsw info my-docs
```

### 특정 문서 조회
```bash
v-hnsw get my-docs 42 17 100
```

### 문서 삭제
```bash
v-hnsw delete my-docs --id 42
```

### 내보내기/가져오기
```bash
v-hnsw export my-docs -o backup.jsonl
v-hnsw import my-docs -i backup.jsonl
```

---

## 5. 권장 설정

### RTX 3060 6GB 환경

| 설정 | 값 | 이유 |
|------|-----|------|
| 모델 | `bge-base-en-v1.5` | 768차원, 110MB, 좋은 품질 |
| 배치 크기 | `128` | VRAM 안정적 사용 |
| FP16 | `--fp16` | VRAM 50% 절약 |
| Device | `cuda` | GPU 가속 |

```bash
v-hnsw insert my-docs -i data.jsonl \
  --embed \
  --model bge-base-en-v1.5 \
  --batch-size 128 \
  --device cuda \
  --fp16
```

### CPU 환경

| 설정 | 값 | 이유 |
|------|-----|------|
| 모델 | `all-mini-lm-l6-v2` | 384차원, 22MB, 빠름 |
| 배치 크기 | `256` | CPU에서 효율적 |
| Device | `cpu` (기본값) | - |

```bash
v-hnsw insert my-docs -i data.jsonl \
  --embed \
  --model all-mini-lm-l6-v2 \
  --batch-size 256
```

### 다국어 (한국어 포함)

```bash
v-hnsw insert my-docs -i data.jsonl \
  --embed \
  --model multilingual-e5-base \
  --device cuda
```

---

## 6. JSONL 형식

### 기본 형식 (텍스트만)
```json
{"text": "문서 내용..."}
```

### 메타데이터 포함
```json
{"text": "문서 내용...", "payload": {"title": "제목", "url": "https://..."}}
```

### 벡터 직접 제공 (임베딩 불필요)
```json
{"vector": [0.1, 0.2, ...], "text": "문서 내용...", "payload": {...}}
```

---

## 7. 전체 워크플로우 예시

```bash
# 1. 마크다운 폴더를 JSONL로 변환
./md-to-jsonl.sh ~/Documents/notes > notes.jsonl

# 2. 인덱싱 (GPU 사용)
v-hnsw insert notes-db -i notes.jsonl --embed --model bge-base-en-v1.5 --device cuda --fp16

# 3. 인덱스 빌드
v-hnsw build-index notes-db

# 4. 검색
v-hnsw vsearch notes-db "rust 에러 핸들링 방법" --show-text

# 5. 결과 확인
v-hnsw get notes-db 42
```

---

## 문제 해결

### "Model dimension doesn't match database dimension"
- 이미 다른 모델로 인덱싱된 데이터베이스입니다
- 새 데이터베이스를 만들거나 기존 것을 삭제하세요

### "HNSW index not found"
- `v-hnsw build-index <db>` 실행 필요

### CUDA 오류
- CUDA Toolkit 설치 확인
- `--device cpu`로 CPU 모드 사용

### 메모리 부족
- `--batch-size`를 줄이세요 (64, 32 등)
- `--fp16` 옵션 사용
