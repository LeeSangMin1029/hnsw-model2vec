# 002: FieldNorm 256-entry 캐시 ✅ 완료

## 구현 요약
- `fieldnorm.rs`: encode(doc_len→u8), decode(u8→f32), `FieldNormLut` (256-entry 역수 LUT)
- `index.rs`: `fieldnorm_codes: HashMap<PointId, u8>` + `fieldnorm_lut: Option<FieldNormLut>` 추가
  - `add_document()`: 코드 생성, LUT 무효화
  - `build_fieldnorm_cache()`: 벌크 삽입 후 명시 호출
  - `load()`/`load_mutable()`: 로드 시 자동 캐시 빌드
  - `score_posting()`: LUT 있으면 O(1) 조회, 없으면 기존 fallback
- `maxscore.rs`: `fieldnorm_lut`/`fieldnorm_codes` 파라미터 추가
- 빌드 파이프라인: `buildindex.rs`, `indexing.rs` (full+incremental) 모두 save 전 캐시 빌드

## 검증
- 1326 테스트 전체 통과 (기존 1315 + fieldnorm 11개)
- 양자화 오차: doc_len 1~10000에서 score 오차 < 5%
- 랭킹 순서 보존 확인 (lut_scoring_preserves_ranking 테스트)
- clippy 0 warnings
