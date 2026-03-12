# v-hnsw / v-code

## 크레이트 구조

```
바이너리:
  v-hnsw          ← 문서 검색 CLI (markdown, JSONL)
  v-code          ← 코드 인텔리전스 CLI (구조 분석, 클론 감지, 코드 검색)

공유 인프라 (v-hnsw-cli lib):
  db_config       ← DbConfig (공유 설정)
  ingest          ← IngestRecord, make_payload, truncate_for_embed, CodeChunkEntry, build_called_by_index, code_chunk_to_record
  search_result   ← FindOutput, compact_output, print_find_output, SearchResultItem
  file_utils      ← generate_id, normalize_source, scan_files
  file_index      ← 파일 메타데이터 인덱스

코드 전용:
  v-code-chunk    ← tree-sitter 기반 코드 청킹 (Rust/TS/Python/Go/Java/C/C++)
  v-code-intel    ← 호출 그래프, 영향 분석, 클론 감지, reasoning

검색 엔진:
  v-hnsw-core     ← VectorIndex trait, PayloadStore trait
  v-hnsw-graph    ← HNSW 그래프 구현
  v-hnsw-search   ← BM25 + hybrid search + 토크나이저
  v-hnsw-storage  ← StorageEngine, mmap 벡터, WAL
  v-hnsw-embed    ← Model2Vec 임베딩
  v-hnsw-rerank   ← cross-encoder reranking (ms-marco-TinyBERT)
```

## 금지 사항

- **Explore 에이전트 절대 사용 금지**: `subagent_type="Explore"` 사용하지 않음. 직접 Glob/Grep/Read 또는 v-code 사용.
- **코드 검색은 `v-code` 사용**: 코드베이스 검색에는 `v-code` code-intel 커맨드 사용. `v-hnsw find`는 자연어/문서 퍼지 검색 전용.
  - 구조 분석: `def`, `refs`, `symbols`, `stats`
  - 그래프 탐색: `impact` (callers BFS), `gather` (callees+callers 통합), `trace` (경로)
  - 코드 검색: `find` (hybrid BM25+HNSW + cross-encoder reranking)
  - Reasoning: `detail` (설계 판단/이력 조회·추가)
  - 클론 감지: `dupes` (AST hash, MinHash Jaccard)
  - `--include-tests` 로 테스트 포함, `--detail` 로 reasoning 포함

## 에이전트 스폰 규칙

- **모든 에이전트 프롬프트에 v-code 사용 지시 포함 필수**. 아래 블록을 에이전트 프롬프트 끝에 항상 추가:
  ```
  ## v-code 코드 검색 (필수)
  코드베이스 탐색 시 전체 파일을 읽지 말고 v-code code-intel 커맨드를 사용해라.
  DB 경로: .v-hnsw-code.db
  - 심볼 정의: v-code def <db> <symbol>
  - 참조 검색: v-code refs <db> <symbol>
  - 호출 그래프: v-code gather <db> <symbol> --depth 2
  - 영향 분석: v-code impact <db> <symbol> --depth 2
  - 전체 파일 대신 줄 범위만 읽어라: Read file_path offset=N limit=M
  ```

## Rust 코딩 규칙

- `&T` 우선, `.clone()` 최소화. payload-less enum → `Copy`
- `Result<T, E>` + thiserror `#[from]`. `unwrap()` 금지
- `#[expect(clippy::lint)]` > `#[allow]`
- iterator 우선, 루프 내 `.clone()` 금지
- imports: std → 외부 → workspace → crate
- clippy `all=deny` + `pedantic=warn`, `unsafe_code="warn"`

## 테스트 구조

- 소스 파일에 테스트 코드 0줄 — `lib.rs/main.rs` → `tests/mod.rs` 중앙 참조
- 서브모듈은 자체 `tests/` 보유, `#[path]`로 연결
- 테스트에서 private 접근 필요 시 `pub(crate)` 사용
