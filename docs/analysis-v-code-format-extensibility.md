# v-code 출력 포맷 분석 + 다국어 확장 가능성 (2026-03-23)

## 출력 포맷 일관성

### 일관된 것
- 파일 경로: `[A]file.rs` alias 형식 — 모든 명령어 통일
- 라인 번호: `:start-end` — `format_lines_opt()` 통일
- kind 태그: function 생략, struct/enum/trait 표시 — 통일
- call_site: `→ :line` — context에서만 (적절)

### 불일치
- **헤더**: context/blast/coverage는 `=== xxx ===`, stats/symbols/trace는 없음
- **[test] 마커**: context/blast에서만 표시, jump/trace에서 누락
- **jump 헤더**: `Execution Flow` (대문자) — 다른 명령어는 소문자

### 중복
- stats와 coverage의 crate 테이블에 prod_fn/test_fn 중복 — 맥락이 달라 수용
- context callers와 blast — depth 다름, 목적 다름 — 수용

### 개선 후보
1. 헤더 `=== xxx ===` 통일 (stats, symbols, trace에 추가)
2. jump/trace에 [test] 마커 추가
3. alias legend 출력 (현재 global_aliases()의 legend 미사용)

---

## 다국어 확장 가능성

### 아키텍처 분류

| 계층 | Rust 종속 | 범용 |
|------|----------|------|
| **extractor** | mir-callgraph (376줄) | - |
| **JSONL 포맷** | - | edges.jsonl + chunks.jsonl |
| **edge_resolve** | closure strip (10줄) | 나머지 전부 |
| **graph.rs** | - | 완전 범용 |
| **ParsedChunk** | - | 완전 범용 |
| **chunk_from_mir** | import/test 파싱 (~30줄) | 나머지 전부 |

### 확장 용이성 (언어별)

| 언어 | 난이도 | 도구 | 비고 |
|------|--------|------|------|
| **TypeScript** | 낮음 | TS compiler API | type-aware, API 잘 설계됨 |
| **Go** | 낮음 | go/callgraph (CHA/VTA/RTA) | 표준 라이브러리 내장 |
| **Python** | 중간 | pyright/mypy | dynamic dispatch로 정확도 제한 |
| **Java** | 높음 | Soot/WALA | bytecode 필요, 도구 무거움 |
| **C/C++** | 높음 | clang AST/LLVM IR | 매크로/템플릿 복잡성 |

### 확장 방법

```
현재: mir-callgraph → edges.jsonl + chunks.jsonl → v-code
확장: ts-callgraph  → edges.jsonl + chunks.jsonl → v-code (동일 파이프라인)
      go-callgraph  → edges.jsonl + chunks.jsonl → v-code (동일 파이프라인)
```

JSONL 스키마가 언어 중립적이므로, 언어별 extractor만 만들면 graph.rs/edge_resolve/coverage 전부 무수정 동작.
수정 필요: chunk_from_mir의 Rust 문법 하드코딩 (~30줄) → 언어별 전략 패턴
