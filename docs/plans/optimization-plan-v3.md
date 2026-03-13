# 최적화 계획 v3 (2026-03-14)

## 1. v-hnsw에서 tree-sitter 제거 [치명적/빌드+크기]
- `v-hnsw-cli`가 `v-code-chunk`에 의존 → 문서 전용 바이너리에 코드 파서 포함
- 해결: `v-code-chunk` 의존을 feature gate (`code` feature)로 분리
- 효과: v-hnsw 바이너리 크기 감소, 빌드 시간 단축

## 2. BM25 3중 저장 제거 [치명적/디스크]
- bm25.bin(3.4M) + bm25.snap(5.3M) + bm25_data.bin(3.1M) = 11.8MB
- snap이 있으면 bin 불필요. FST data는 snap에 포함
- 해결: snap+fst를 primary로, bincode 제거 (마이그레이션 로직 추가)

## 3. HNSW 노드 HashMap → Vec [성능]
- `nodes: HashMap<PointId, Node>` → 검색 핫패스마다 hash lookup
- PointId가 연속적이면 `Vec<Option<Node>>`로 O(1) 접근
- 해결: 연속 ID 보장 확인 후 Vec 기반으로 전환

## 4. v-hnsw find 출력 토큰 최적화 [토큰]
- 기본 JSON 출력이 AI 에이전트에 토큰 낭비
- 해결: `--compact` 또는 text 기본 출력 모드 추가

## 5. 빌드 시간 최적화 [빌드]
- 1번 해결 시 자동으로 개선 (tree-sitter C 컴파일 제거)
- 추가: 의존성 정리, 불필요한 feature 제거

## 우선순위
1번 → 5번(1번에 포함) → 4번 → 2번 → 3번
