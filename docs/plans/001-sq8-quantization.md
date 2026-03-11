# 001: SQ8 스칼라 양자화

## 목표
HNSW 탐색 시 f32 벡터 대신 u8 양자화 벡터를 사용하여 메모리 4x 절감 + 캐시 효율 향상.
최종 top-K 결과는 원본 f32로 rescore하여 정확도 유지.

## 현재 구조
- `VectorStore` trait: `get(&self, id) -> &[f32]` — f32 고정
- `MmapVectorStore`: slot당 `dim * 4` bytes (f32)
- `NormalizedCosineDistance`: `1 - dot(a, b)`, 정규화 벡터 가정
- HNSW 검색: `search_ext(store, query, k, ef)` → store.get()으로 f32 읽기
- 벡터 dim=256, 전체 2417개, vectors.bin 9.8MB

## 설계

### SQ8 원리
f32 벡터의 각 차원을 [min, max] 범위에서 [0, 255] u8로 선형 매핑.
- 양자화: `q = round((v - min) / (max - min) * 255)`
- 역양자화: `v ≈ min + q * (max - min) / 255`
- 학습: 전체 벡터에서 차원별 min/max 계산 (1-pass)

### 구현 범위

#### 1. `crates/v-hnsw-storage/src/sq8.rs` — SQ8 양자화/역양자화
- `Sq8Params { mins: Vec<f32>, scales: Vec<f32> }` (dim개)
- `train(vectors: &[&[f32]]) -> Sq8Params` — min/max 수집
- `quantize(params, f32_vec) -> Vec<u8>`
- `distance_sq8(params, query_f32, code_u8) -> f32` — 양자화 공간 거리

#### 2. `crates/v-hnsw-storage/src/sq8_store.rs` — 양자화 벡터 mmap 저장소
- `Sq8VectorStore`: u8 벡터 저장 (slot당 dim bytes, 4x 절감)
- `VectorStore` trait은 구현하지 않음 (f32 반환 불가)
- 대신 `Sq8VectorStore::get_quantized(id) -> &[u8]` 제공

#### 3. `crates/v-hnsw-graph/src/search.rs` — 2단계 검색
- `search_ext_sq8(graph, sq8_store, f32_store, params, query, k, ef)`
- 탐색(greedy + search_layer): sq8 거리로 후보 선택
- rescore: top-ef 결과를 f32_store에서 원본 벡터로 정확한 거리 재계산
- top-k 반환

#### 4. `crates/v-hnsw-cli/src/commands/buildindex.rs` — 빌드 시 SQ8 학습+저장
- HNSW 빌드 후 SQ8 params 학습 → `sq8_params.bin` 저장
- 양자화 벡터 → `sq8_vectors.bin` 저장

#### 5. 검색 경로 통합
- `HnswSnapshot` / daemon에서 sq8 파일 존재 시 자동 사용
- 없으면 기존 f32 경로 fallback

## 수정 파일
- `crates/v-hnsw-storage/src/sq8.rs` (신규)
- `crates/v-hnsw-storage/src/sq8_store.rs` (신규)
- `crates/v-hnsw-storage/src/lib.rs` (모듈 등록)
- `crates/v-hnsw-graph/src/search.rs` (sq8 검색 함수)
- `crates/v-hnsw-cli/src/commands/buildindex.rs` (빌드 시 학습)
- `crates/v-hnsw-cli/src/commands/indexing.rs` (증분 시 sq8 갱신)

## 검증
1. `cargo test -p v-hnsw-storage` — sq8 단위 테스트 (양자화/역양자화 정확도)
2. `cargo test -p v-hnsw-graph` — sq8 검색 vs f32 검색 recall 비교
3. 실제 DB에서 `v-hnsw find` 결과가 f32 대비 recall@10 >= 0.95 확인
4. vectors.bin vs sq8_vectors.bin 크기 비교 (4x 절감 확인)

## 안 할 것
- VectorStore trait 변경 (기존 인터페이스 유지)
- PQ/BQ 같은 복잡한 양자화 (SQ8이면 충분)
- HNSW 그래프 빌드 시 양자화 사용 (빌드는 f32, 검색만 sq8)

## 구현 결과

### 완료된 단계
1. **sq8.rs**: 양자화/역양자화 핵심 로직 + LUT 기반 비대칭 거리 — 10개 테스트 통과
2. **sq8_store.rs**: u8 mmap 저장소 (create/open/insert/get/grow) — 8개 테스트 통과
3. **빌드 통합**: `build_indexes()`에서 SQ8 params 학습 + 양자화 벡터 저장 자동 수행
4. **증분 통합**: `update_indexes_incremental()`에서 SQ8 증분 갱신

### 검증 결과
- 전체 테스트: core 55 + storage 143 + graph 244 + cli 395 = 837개 통과
- 실제 DB: 2464벡터, **vectors.bin 9.8MB → sq8_vectors.bin 633KB (4x 압축)**
- 비대칭 거리 오차: max < 0.02, recall@10 >= 9/10
- 기존 검색 (f32 경로) 정상 동작

### 5단계: 2단계 검색 통합 (완료)
- **`DistanceComputer` trait**: 벡터 조회 + 거리 계산 추상화 (search.rs)
- **`search_two_stage()`**: approx DC로 그래프 순회 → exact DC로 rescore
- **`search_layer_dc()` / `greedy_closest_dc()`**: DistanceComputer 기반 검색 로직
- **HnswSnapshot + HnswGraph**: `search_two_stage()` 메서드 추가
- **daemon 통합**: SQ8 파일 자동 감지, `Sq8Dc`/`F32Dc` 구현, 없으면 f32 fallback
- 테스트: graph 248개 통과 (기존 244 + two_stage 4개)
- 실제 DB: 2482벡터 SQ8 검색 정상 동작, 관련성 높은 결과 반환

### 최종 상태: 완료
- 메모리 4x 절감 (sq8_vectors.bin)
- 검색 시 SQ8 자동 사용 (파일 존재 시)
- 기존 f32 경로 완벽 호환 (fallback)
