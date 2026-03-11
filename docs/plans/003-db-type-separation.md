# 003: 문서/코드 DB 빌드 분리

## 목표
`DbConfig`에 DB 타입을 명시하여 문서 DB와 코드 DB 각각에 특화된 빌드 파이프라인 적용.

## 현재 상태
- `detect_input_type()` → 마크다운/코드 자동 감지는 이미 동작
- 인덱싱(`build_indexes`)은 타입 무관하게 동일 경로
- BM25 토크나이저가 코드에도 한국어 형태소 분석 적용 (비효율)

## 설계
```
DbConfig { db_type: "document" | "code" | "auto" }
```

### 문서 DB 전용
- BM25: `KoreanBm25Tokenizer` (형태소 분석)
- TagIndex (Roaming Bitmap 필터)
- FSST 텍스트 압축
- FieldNorm 캐시 (#2 완료)

### 코드 DB 전용
- BM25: `CodeTokenizer` (camelCase/snake_case 분리, 신규)
- SymbolIndex + CallGraph (tree-sitter)
- 클론 감지 (dupes)
- AST 해시

### 공유 (양쪽 동일)
- HNSW + SQ8, StorageEngine, model2vec 임베딩, 데몬, FileIndex

## 수정 파일
- `crates/v-hnsw-cli/src/commands/create.rs` — `DbConfig`에 `db_type` 추가
- `crates/v-hnsw-cli/src/commands/indexing.rs` — 타입별 빌드 분기
- `crates/v-hnsw-search/src/tokenizer/code.rs` (신규) — 코드 토크나이저
- `crates/v-hnsw-cli/src/commands/add/` — auto 감지 시 db_type 설정

## 선행 작업
- [x] #1 SQ8 양자화
- [x] #2 FieldNorm 캐시
- [ ] #3 IDF pre-computation
- [x] #4 CodeTokenizer 구현
