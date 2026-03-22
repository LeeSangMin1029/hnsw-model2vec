# 타입 추론 메모리 문제 조사 (2026-03-23)

## 문제 정의
Rust call graph 구축 시 `infer_body` → salsa 캐시 무한 누적 → OOM.
qdrant (301K lines) 기준 750파일에서 크래시.

## 실측 결과

### 소수 함수 outgoing_calls (qdrant)
| 대상 | 시간 | Commit 변화 |
|------|------|------------|
| Light (ext=0) | 5-57ms | +0~1.7MB |
| Heavy (ext=9) | 105-1280ms | -243~0MB |
| Cached 재호출 | 0.7-3.4ms | +0MB |
| **증분 1파일** | **26ms** | **+0.1MB** |

### 전체 chunk_files (qdrant, 1053파일)
- 750파일에서 **OOM 크래시** (exit code 1)
- RA baseline 4.5GB + outgoing_calls 누적 → 10GB+

## 근본 원인: salsa LRU 비활성

```
InferenceResult::for_body  → #[salsa::tracked] — LRU 옵션 없음
update_base_query_lru_capacities() → 전부 주석 처리 (salsa-transition)
trigger_garbage_collection() → LRU capacity=0이라 evict 대상 없음
```
- salsa 0.25.2 (RA 현재): tracked fn에 LRU 불가
- salsa 0.26.0: tracked LRU 추가, 하지만 lock-congestion 이슈 (#1032)

## SCIP 분석 결과
- 77,263 occurrences, 5,558 unique symbols
- 내부 4,043 / 외부 1,207 — 패키지명으로 즉시 분류
- 동명 함수 113개 완벽 disambiguation (impl#[Type]method)
- 로컬 변수 타입 10,506개 (sig_doc에서 추출)
- **외부 메서드 return type은 SCIP에 없음** (외부 SymbolInformation 3개뿐)

## 검토한 접근법

### 1. SCIP 초기 + RA 증분 (현실적, 채택)
- 초기: `rust-analyzer scip` (별도 프로세스) → 100% 정확
- 증분: daemon outgoing_calls (1~2파일, 26ms, +0.1MB) → 100% 정확
- 단점: 이중 시스템, SCIP 재생성 불가

### 2. 외부 타입 테이블 + forward propagation
- (receiver_type, method) → symbol 매핑 1,123개 SCIP에서 추출 가능
- 외부 return type 없음 → passthrough 규칙(~50개)으로 부분 해결
- **bidirectional inference 필요 패턴 7.6%** (collect, into, parse, closure 등)

### 3. JARVIS 스타일 함수별 type graph
- 함수마다 독립 type graph → 처리 후 해제 → 메모리 누적 없음
- forward propagation으로 92% 해결
- 나머지 8%는 경량으로 불가 (bidirectional inference 필요)

### 4. rustc MIR 직접 추출
- 100% 정확 (컴파일러), incremental compilation으로 증분
- 메모리: 프로세스 종료 시 해제, 디스크 캐시
- 단점: nightly 필요, rustc_private API 불안정, 증분 2-10초

### 5. salsa fork (LRU 활성화)
- InferenceResult::for_body에 lru(N) 추가
- 근본 해결이지만 RA fork + salsa 0.26 lock-congestion 해결 필요

## 관련 논문

| 논문 | 핵심 아이디어 | 적용 가능성 |
|------|-------------|------------|
| JARVIS (2023) | 함수별 type graph, forward propagation | 높음 — 경량 추론 |
| Scalable CHA Stitching (2021) | 외부 lib call graph 캐시 + stitching | 높음 — 외부 타입 테이블 |
| NoCFG (2021) | coarse abstraction, 90% precision | 중간 — leaf type |
| APAK (2026) | context-sensitive, CHA 20%→2% FP | 중간 — trait dispatch |
| Demanded Summarization (TOPLAS 2024) | on-demand + incremental + compositional | 높음 — 이론적 |
| Frankenstein (EMSE 2023) | dependency CG 캐시, F1=0.98 | 높음 — 외부 타입 |
| Incremental CG Industrial (ICSE-SEIP 2023) | 20x 속도, 58% 메모리 | 높음 — 증분 |

## 결론
"타입 추론 → 메모리 폭발" 순환을 끊는 현실적 방법:
1. **당장**: SCIP(초기) + RA outgoing_calls(증분 1~2파일)
2. **중기**: JARVIS 스타일 경량 type graph (92% 커버) + RA fallback
3. **장기**: rustc MIR 기반 또는 salsa LRU 해결
