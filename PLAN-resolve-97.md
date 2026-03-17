# Call Resolution 97%+ 달성 계획

## 현황 (Phase 1 완료)
- Precision: 100% | Recall: 94.5% (rust-analyzer 56K calls, 3091 unresolved)
- 자체 프로젝트: P=100% R=96.2%
- 카테고리: receiver.method 1967, self.field.method 627, bare 306, Type::method 150, self.method 41

## 완료
- [x] Phase 1a: Monomorphism pre-filter (bare 1141→306, -835건)
- [x] Phase 1b: Co-call constraint intersection (receiver/self.method 일부 해소)

## 근본 한계 (100% 불가능한 원인)
- 매크로 생성 함수: salsa의 `#[interned]` → `lookup()` 등 소스에 없음
- dyn Trait 동적 디스패치: 런타임 결정
- 외부 crate 내부 함수: 인덱스에 없음

## Phase 2: Interprocedural Type Propagation (+2~3%)
caller→callee 간 타입 전파. 현재는 함수 내부(intraprocedural)만 추적.

### 2a. Caller→Callee generic instantiation
`process(my_function_id)` 호출 시 `impl Lookup` → `FunctionId`로 치환.
caller의 argument type을 callee의 param type에 매핑 → 제네릭 구체화.
`graph.rs` iterative pass에서 resolved callee의 param_types → caller arg types 역전파 강화.

### 2b. Return type chain 강화
`let y = x.foo(); y.bar()` — foo resolve 시 return type → y 타입 → bar resolve.
현재 iterative pass에서 하지만, resolve 즉시 같은 pass 내에서 재활용하도록 개선.

### 2c. self.field.method 강화
owner_field_types에서 필드 타입 가져오고, 외부 타입이면 extern index에서 method 확인.
`self.flags.contains` → flags: FxHashSet → extern_index.has_method("fxhashset", "contains").

## Phase 3: Co-occurrence PMI scoring (+0.5~1%)
resolved call graph에서 co-occurrence matrix → ambiguous call의 candidate ranking.

## 예상 최종: 97~98% (나머지 2~3%는 매크로/dyn/외부 crate 한계)

## 수정 파일
- `graph.rs`: iterative pass 강화 (2a, 2b), self.field extern fallback (2c)
- `verify.rs`: 변경 없음
