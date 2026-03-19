# dyn Trait receiver resolve 계획

## 현황 (2026-03-19)
- rust-analyzer: P 100%, R 91.3% (unresolved 5,879건)
- 최대 원인: `db: &dyn DefDatabase` → `db.function_signature()` 미연결

## Phase 1: dyn/impl Trait param → receiver_types ✅ 완료

구현 내용:
- generic trait bound 파싱: `fn foo<N: NodeGraph>` → `n → nodegraph`
- trait receiver short_fn fallback (salsa/macro-generated DB methods)
- rustdoc overlay: +40k return types, +14k method owners
- trait_fan_out: receiver.method()만 fan out (Type::method 제외)
- classify_extern_calls에 infer_local_types_from_calls 추가
- `v-code rustdoc <DB>` 서브커맨드 (병렬 생성, 캐시 스킵)
- verify 3-category (Confirmed/Unverified/Wrong)

결과: R 90.8% → 91.3% (+0.5%), P 100% 유지

## Phase 1.5: 글로벌 extern 캐시 공유 ← 진행중

문제: 프로젝트마다 std + deps를 중복 파싱 (1~8MB/프로젝트)
설계:
- `~/.cache/v-code/extern/std-{rustc_hash}.bin` — std 전용
- `~/.cache/v-code/extern/{crate}-{version}.bin` — dep 크레이트별
- build 시 글로벌 캐시 히트 → 파싱 스킵, 미스만 파싱 후 저장

## Phase 2: trait method qualified 호출

`TestDB::with_position()` → `WithFixture::with_position` 연결.
trait impl에서 `for Type` 매핑 활용.
unresolved `Type::method (qualified)`: 1,088건

## Phase 3: nested function / bare call

함수 내부 `fn go()` → 현재 chunk 추출 안됨.
bare call `declare()` → self type의 method로 resolve 시도.
unresolved `bare function`: 183건

## Unresolved 분포 (rust-analyzer)
- receiver.method: 4,257건 (72%)
- Type::method (qualified): 1,088건 (19%)
- self.field.method: 250건
- bare function: 183건
- self.method: 101건
