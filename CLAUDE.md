## Rust 코딩 규칙

- `&T` 우선, `.clone()` 최소화. payload-less enum → `Copy`
- `Result<T, E>` + thiserror `#[from]`. `unwrap()` 금지
- `#[expect(clippy::lint)]` > `#[allow]`
- iterator 우선, 루프 내 `.clone()` 금지
- imports: std → 외부 → workspace → crate
- clippy `all=deny` + `pedantic=warn`, `unsafe_code="warn"`

## 테스트 구조

- **테스트 러너: `cargo nextest run`** (`cargo test` 사용 금지)
- 소스 파일에 테스트 코드 0줄 — `lib.rs/main.rs` → `tests/mod.rs` 중앙 참조
- 서브모듈은 자체 `tests/` 보유, `#[path]`로 연결
- 테스트에서 private 접근 필요 시 `pub(crate)` 사용
