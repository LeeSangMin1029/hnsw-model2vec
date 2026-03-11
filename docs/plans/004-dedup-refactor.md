# 004: 프로덕션 코드 중복 제거 리팩터링 ✅ 완료

## 결과 요약
dupes 분석에서 발견된 5건 중 고가치 2건 완료, 3건은 이미 해결/비효과적으로 건너뜀.

### 1. `build_sq8()` 통합 ✅
- `buildindex.rs` 중복 함수 삭제 → `indexing::build_sq8()` 호출
- capacity 버그 수정: `vectors.len() + 64` → `max_slot + 1`

### 2. BM25 스코어링 공통화 ✅
- `scorer.rs`에 `ScoringCtx`, `PostingView` trait, `accumulate_and_rank()`, `score_documents_common()` 추출
- `Bm25Snapshot`에 FieldNorm LUT 추가 (기존 미적용 → O(1) 조회)
- Net -57 lines

### 3. CodeChunker 공통 베이스 — 이미 해결됨
- `define_chunker!` 매크로 + `LangExtractors` 테이블이 이미 공통 베이스 역할
- Go/C만 언어 고유 로직으로 별도 구현 (정당한 차이)

### 4. `require_exists/not_exists` — 건너뜀
- `require(name, should_exist: bool)` 병합은 boolean param anti-pattern
- 7줄짜리 명확한 이름의 함수 2개가 더 가독성 좋음

### 5. `load()` 보일러플레이트 — 건너뜀
- 에러 타입(anyhow vs VhnswError), missing-file 처리가 각각 다름
- 공통화하면 오히려 복잡도 증가
