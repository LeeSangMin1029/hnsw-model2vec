# 에이전트 코드 검색 전략 벤치마크

## 테스트 조건

- **프롬프트**: `build_called_by_index` 함수의 구현 위치, 파라미터/반환값, 호출 위치 분석
- **프로젝트**: v-hnsw (Rust, ~40 크레이트, ~1300 테스트)
- **측정일**: 2026-03-13

## Claude Code (Opus) 전략 비교

| 전략 | 호출수 | 토큰 | 시간 |
|---|---|---|---|
| C: Grep only | 7 | 17.4K | 39s |
| F: Grep→v-code | 7 | 19.2K | 47s |
| E: v-code→Grep | 16 | 21.2K | 69s |
| A: v-code only | 9 | 22.5K | 44s |
| D: v-code+v-hnsw | 9 | 25.8K | 55s |
| B: v-hnsw only | 7 | 37.9K | 50s |

## pi Haiku 전략 비교 (정밀 토큰 측정)

| 전략 | API 호출 | 도구 호출 | Input | Output | 총 토큰 (비캐시) | 시간 |
|---|---|---|---|---|---|---|
| v-code 스킬 | 8 | 7 | 3,694 | 2,381 | **6,075** | 26s |
| grep→v-code 스킬 | 12 | 11 | 3,811 | 2,891 | **6,702** | 37s |
| grep only | 6 | 5 | 5,673 | 1,840 | **7,513** | 19s |
| 스킬 없음 | 6 | 5 | 8,967 | 2,255 | **11,222** | 31s |

## 결론

### 용도별 최적 전략

| 목적 | 최적 전략 | 이유 |
|---|---|---|
| 단순 위치 찾기 | **Grep only** | 최소 호출, 최단 시간 |
| 호출 관계 분석 | **v-code 스킬** | 구조화된 출력, 최소 토큰 |
| 범용 (위치+관계) | **Grep → v-code** | Grep으로 빠르게 위치 확인 후 v-code로 심화 분석 |

### 핵심 인사이트

1. **스킬 적용 시 비캐시 토큰 46% 감소** (11.2K → 6.1K)
2. **v-code 구조화 출력이 토큰 효율의 핵심** — grep 결과는 raw 텍스트라 토큰 소모 큼
3. **grep→v-code 혼합은 오버헤드만 증가** — 도구 호출 11회, 이점 없음
4. **v-hnsw 시맨틱 검색은 코드 위치 찾기에 비효율** — 정확한 심볼명 알면 grep/v-code가 우월

### 에이전트 검색 전략 권장

```
1단계: 심볼명을 아는 경우
  → grep -rn "<symbol>" --include="*.rs" (또는 Grep 도구)
  → 위치 확인 후 Read offset=N limit=M 으로 줄 범위만 읽기

2단계: 호출 관계/영향 분석 필요 시
  → v-code def <db> <symbol>       (정의)
  → v-code refs <db> <symbol>      (참조)
  → v-code impact <db> <symbol>    (callers BFS)
  → v-code gather <db> <symbol>    (callees + callers)

3단계: 심볼명 모를 때
  → v-code symbols <db> -n <키워드> --compact --limit 20
  → 또는 v-hnsw find <db> "<자연어 설명>" -k 5
```
