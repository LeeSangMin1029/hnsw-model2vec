# v-code verify 100% — 남은 5개 패턴 해결 계획

## 현황 (P1+R1 완료 후)

| 지표 | 자체 (1.6K) | rust-analyzer (22K) |
|---|---|---|
| Precision | 100% | 100.0% (13 wrong) |
| Recall | 99.8% (16) | 97.5% (1383 unresolved) |

## 미해결 5개 패턴

### 1. receiver 타입 미추론 (615건, 44%)
```rust
let x = some_complex_expr();
x.krate()  // x의 타입 불명 → krate 매칭 불가
```
**해결**: `graph.rs` resolve_chunk_edges — multi-hop 반환 타입 전파
- 현재: `let x = foo()` → foo 반환타입으로 x 타입 추론 (1-hop, 2nd pass)
- 개선: iterative fixpoint — 새 타입이 추론될 때까지 반복 (최대 3회)
- `let db = self.db()` → db:Db → `db.generic_params()` → Db::generic_params
- **파일**: `graph.rs` build_with_rustdoc 2nd pass → iterative pass
- **영향**: 615 → ~200 (추정 67% 해소)
- **범용성**: ★★★★★ 언어 무관, 모든 Rust 프로젝트 동작

### 2. 매크로 호출 (bare ~30건, 2%)
```rust
assert_eq!(a, b);  // tree-sitter: "assert_eq" 함수 호출로 추출
```
**해결**: `extract/common.rs` — macro_invocation node 감지
- tree-sitter Rust에서 매크로 호출은 `macro_invocation` node type
- `extract_callee_from_node`에서 `macro_invocation` → skip
- 또는 `!` 포함 여부로 매크로 판별 (tree-sitter가 `!` 보존)
- **파일**: `extract/common.rs` walk_for_calls_with_lines
- **영향**: bare function 460 → ~430
- **범용성**: ★★★★★ tree-sitter 문법 기반

### 3. 외부 crate qualified 호출 (105건, 8%)
```rust
Command::new("test")              // std::process::Command
lsp_server::Response::new_ok()    // 외부 crate
```
**해결**: `graph.rs` + `verify.rs` — cargo dep extern index 확장
- 현재 extern_index가 direct dep만 파싱 — transitive dep 미포함
- `lsp_server`는 transitive dep → extern index에 없음
- **구현**: discover_cargo_deps에서 Cargo.lock 기반 전체 dep 파싱
- **파일**: `extern_types.rs` discover_cargo_deps
- **영향**: 105 → ~30
- **범용성**: ★★★★☆ Cargo 프로젝트 전용 (대부분의 Rust)

### 4. self.field 체인 (106건, 8%)
```rust
self.output.push_str("x")  // output: String이지만 제네릭 추론 못함
self.map.get(&key)          // map: HashMap<K,V> → leaf "hashmap" 추론 필요
```
**해결**: `graph.rs` resolve_chunk_edges — 제네릭 필드 타입 전파
- 현재: owner_field_types에서 `output: string` 추출은 가능
- 문제: `push_str`이 extern index의 `string::push_str`로 매칭되어야 하는데 안 됨
- **구현**: self.field.method에서 field 타입 → extern_index.has_method 체크 추가
- **파일**: `graph.rs` resolve_with_imports (self.field.method 분기)
- **영향**: 106 → ~30
- **범용성**: ★★★★★ struct 정의에서 필드 타입 추출, 언어 무관

### 5. trait impl self 타입 (97건, 7%)
```rust
impl SomeTrait for ConcreteType {
    fn method(&self) {
        self.concrete_method()  // self = ConcreteType
    }
}
```
**해결**: `graph.rs` ChunkMeta — trait impl의 concrete type 추출
- 현재 owning_type이 `impl` chunk에서 첫 번째 타입만 추출
- trait impl: `impl Trait for Type` → Type이 concrete, Trait은 아님
- `impl<'db> InferenceContext<'db>` 같은 경우 이미 작동 — 문제는 trait impl
- **구현**: loader/parse에서 impl chunk의 `for` 뒤 타입을 ParsedChunk에 저장
- **파일**: `parse.rs` ParsedChunk + `extract/chunk.rs` impl 파싱
- **영향**: 97 → ~20
- **범용성**: ★★★★★ Rust trait impl 문법 기반

## 구현 순서 (영향도 순)

| 순서 | 패턴 | 예상 해소 | 누적 recall |
|---|---|---|---|
| 1 | receiver 타입 iterative 전파 | ~400건 | 98.2% |
| 2 | self.field extern 매칭 | ~75건 | 98.7% |
| 3 | trait impl self 타입 | ~75건 | 99.2% |
| 4 | 외부 crate dep 확장 | ~75건 | 99.7% |
| 5 | 매크로 호출 skip | ~30건 | 99.9% |

## 범용성 보장 원칙

- **hardcode 금지**: std 타입 목록, 매크로 이름 등 절대 hardcode하지 않음
- **tree-sitter 기반**: AST node type으로 판별 (macro_invocation, primitive_type 등)
- **Cargo 생태계 활용**: Cargo.lock, rustc sysroot, cargo doc으로 자동 탐색
- **DB 자체 정보**: enum_variants, field_types, return_type 등 이미 추출된 메타데이터 활용
- **iterative 추론**: 1회 pass가 아닌 fixpoint 반복으로 정보 전파
