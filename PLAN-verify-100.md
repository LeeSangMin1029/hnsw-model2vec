# v-code verify 100% 정확도 달성 계획

## 현황 (rust-analyzer 22K 심볼, 자체 프로젝트 1.6K)

| 지표 | 자체 프로젝트 | rust-analyzer |
|---|---|---|
| Precision | 100% (0 wrong) | 99.8% (109 wrong) |
| Recall | 99.8% (16 unresolved) | 97.5% (1420 unresolved) |

## Precision Wrong 109개 분석

| 패턴 | 건수 | 원인 | 범용성 |
|---|---|---|---|
| `Cow::Borrowed`/`Cow::Owned` → `TokenText::borrowed`/`owned` | 25 | std enum variant(Cow)가 short fallback에서 프로젝트 함수에 매칭 | 모든 프로젝트 |
| `FunctionBody::node` | 8 | 동명 메서드 ambiguity — `node()`가 여러 타입에 존재 | 대형 프로젝트 |
| `projection`/`tuple`/`infer` | 15 | bare enum variant가 동명 함수에 매칭 (Ty::Tuple 등) | 컴파일러류 |
| `ModPath::is_Self` 등 | 10+ | qualified call이 다른 타입의 동명 메서드에 매칭 | 대형 프로젝트 |
| 나머지 | ~50 | short fallback의 과도한 매칭 | 공통 |

### 수정 P1: std enum variant을 extern으로 분류
**파일**: `graph.rs` resolve_with_imports
- `Cow::Borrowed`, `Cow::Owned` → rustdoc/extern index에서 Cow가 std 타입임을 확인하면 skip
- `Rc::new`, `Arc::new`, `Box::new` 같은 std wrapper도 동일 패턴
- **구현**: resolve_with_imports에서 `Type::Name` 형태일 때, Type이 extern_types에 있으면 skip

### 수정 P2: short fallback에서 enum variant 추가 감지
**파일**: `graph.rs` resolve_with_imports 6번(short fallback)
- 현재: `Type::` prefix면 `exact.get(prefix_leaf)`로 enum 체크
- 개선: prefix가 enum_types에 있으면 skip (exact에 없는 외부 enum도 처리)
- `projection`, `tuple`, `infer` 등 bare call이 enum variant일 때 → 원래 call이 대문자면 skip

### 수정 P3: qualified call의 타입 불일치 감지
**파일**: `graph.rs` resolve_with_imports
- `ModPath::is_Self` 매칭 시 실제 caller의 타입 context와 callee 소속 타입 일치 여부 확인
- 이미 receiver_types 기반 체크가 있지만 qualified call에는 미적용

## Recall Unresolved 1420개 분석

| 카테고리 | 건수 | 상위 예시 | 원인 |
|---|---|---|---|
| receiver.method | 617 | krate(21), adjusted(19), as_expr(16) | receiver 타입 미추론 |
| bare function | 505 | then_some(34), assert_eq(29), text(13) | std 메서드 or 매크로 |
| self.field.method | 106 | self.output.push_str(8), self.interner(7) | field 타입 미추론 |
| self.method | 97 | self.infer_pat(4), self.id.lookup(4) | self 타입 미추론 (trait impl 내) |
| Type::method | 95 | ast::Expr::Literal(6), MemoryMap::default(4) | 외부 타입 or enum variant |

### 수정 R1: std bare method extern 분류 (bare function 505 → ~200)
**파일**: `verify.rs` check_extern_reason, `extern_types.rs`
- `then_some`, `then`, `is_none`, `map_err`, `to_vec`, `clone_for_update` → Option/Result/Iterator 메서드
- `assert_eq` → 매크로 (tree-sitter가 호출로 추출)
- **구현**: extern index에 `Option::then_some`, `bool::then` 등 std trait 메서드 추가
- 또는 verify에서 bare call이 extern_index.methods에도 있으면 extern 분류

### 수정 R2: receiver 타입 전파 강화 (receiver.method 617 → ~300)
**파일**: `graph.rs` resolve_chunk_edges, build_with_rustdoc
- 현재: param_types, local_types, field_types에서 receiver 타입 추론
- 개선: let_call_bindings의 반환 타입 전파 (1-hop chain)
- `let db = self.db()` → db: Database → `db.generic_params()` 해석 가능
- 이미 2nd pass에서 일부 수행, 정확도 개선 여지

### 수정 R3: trait impl self 타입 해석 (self.method 97 → ~30)
**파일**: `graph.rs` resolve_chunk_edges
- trait impl 메서드의 self 타입이 impl 블록의 concrete type
- `impl Foo for Bar { fn go(&self) { self.bar_method() } }` → self: Bar
- 현재 owning_type이 impl 블록에서 추출하지만 trait impl은 미처리

### 수정 R4: enum variant → extern 분류 (Type::method 95 → ~50)
**파일**: `verify.rs`
- `ast::Expr::Literal`, `ast::Adt::Union` 등은 enum variant
- 현재 enum_variant_set에 없는 외부 crate enum (ast crate 등)
- **구현**: `Type::Name` 형태에서 Name이 대문자시작이면 extern 분류 후보

## 모든 Rust 프로젝트 범용성 평가

| 기법 | 범용성 | 근거 |
|---|---|---|
| Prelude variant skip (Ok/Err/Some/None) | ★★★★★ | Rust 언어 사양, 모든 프로젝트 동일 |
| tree-sitter enum variant 추출 | ★★★★★ | 언어 문법 기반, 프로젝트 무관 |
| std extern index | ★★★★☆ | std는 고정, 외부 crate는 rustdoc 의존 |
| receiver 타입 전파 | ★★★★☆ | 패턴 기반 추론, 복잡한 제네릭은 한계 |
| short fallback enum 체크 | ★★★★★ | enum kind 체크는 DB 정보만 사용 |
| trait impl self 해석 | ★★★★☆ | impl 블록 구조 분석, 일반적 패턴 |

**결론**: 모든 수정 사항은 Rust 언어 구조와 std 라이브러리에 기반하므로 프로젝트 특화 로직이 아님.
단, 대형 프로젝트(22K+ 심볼)에서 동명 메서드 충돌이 많아지므로 precision이 더 중요해짐.
궁극적 한계: tree-sitter만으로는 full type inference 불가 → 99.9%+ 가능하나 100%는 이론적 한계.

## 구현 우선순위

1. **P1 + R1**: std enum variant/method extern 분류 → precision +0.1%, recall +2%
2. **P2**: short fallback enum variant 감지 강화 → precision 잔여 wrong 절반 해소
3. **R2**: receiver 타입 전파 → recall +1~2%
4. **R3 + R4**: trait impl self + 외부 enum variant → recall +0.5%
5. **P3**: qualified call 타입 불일치 → precision 최종 정리

## 목표

| 지표 | 현재 | 목표 |
|---|---|---|
| Precision (자체) | 100% | 100% 유지 |
| Precision (RA) | 99.8% | 99.95%+ |
| Recall (자체) | 99.8% | 99.9%+ |
| Recall (RA) | 97.5% | 99%+ |
