# v-code verify 100% — verify 로직 정확성 + recall 개선

## 현황 (iterative 전파 + trait impl + transitive deps 완료 후)

| 지표 | 자체 (1.6K) | rust-analyzer (22K) |
|---|---|---|
| Precision | 100% (0 wrong) | 100.0% (11 wrong) |
| Recall | 99.9% (7 unresolved) | 98.1% (1051 unresolved) |
| Extern 분류 | 502건 | 3428건 |

## Phase 1: verify 로직 정확성 검증 (우선)

### 문제: `check_extern_reason`이 recall을 부풀림

현재 그래프가 resolve 못 한 호출을 "extern이니까 OK"로 분류하는데,
그 중 일부는 그래프가 resolve했어야 할 프로젝트 함수 호출임.

**rust-analyzer extern 3428건 내역:**
- `bare-extern`: 454건 (`to_owned` 122, `iter` 38, `as_ref` 30...)
- `untyped-extern`: 1407건 (`iter` 154, `clone` 100, `map` 87, `syntax` 42, `lookup` 24...)
- `self.field`: 타입 매칭 → 비교적 정확
- `receiver`: 타입 매칭 → 비교적 정확

**수정 1: untyped-extern 엄격화**
- 현재: receiver가 프로젝트 타입이 아니고 method가 extern에 있으면 → extern
- 문제: `syntax`(42건), `lookup`(24건) 같은 프로젝트 함수 이름이 extern으로 숨겨짐
- 수정: method가 **프로젝트 함수 이름과 겹치면** unresolved로 분류
- 단, `len`/`is_empty`/`get`/`push`/`iter`/`clone` 등 std에 압도적인 메서드는 예외
- → 예외 없이: "프로젝트에 같은 이름 함수가 있고 + 그래프가 resolve 안 했으면 = unresolved"

**수정 2: bare-extern 엄격화**
- 현재: bare `len` 호출이 extern에 있으면 → extern
- 문제: 프로젝트에도 `len` 함수가 있는데 resolve 못 한 것일 수 있음
- 수정: 동일하게 프로젝트 함수 겹침 체크

**파일**: `verify.rs` check_extern_reason
**영향**: recall 숫자가 떨어지지만 (98.1% → ~95%), 그게 진짜 숫자

## Phase 2: 진짜 recall 올리기 (Phase 1 이후)

Phase 1에서 드러난 실제 unresolved 분포에 따라 우선순위 재설정.

예상 카테고리:
1. `receiver.method` — 타입 미추론 (가장 많을 것)
2. `bare function` — trait method bare 호출, 매크로 생성 함수
3. `Type::method` — 외부 crate qualified
4. `self.method` — trait dispatch
5. `self.field.method` — 중첩 필드 체인

## 범용성 원칙

- **hardcode 금지**: std 타입/메서드 목록 hardcode 안 함
- **프로젝트 종속 금지**: 특정 프로젝트 패턴에 맞추지 않음
- **verify는 보수적**: 의심스러우면 unresolved로 분류 (recall 부풀리지 않음)
- **extern 분류는 확실한 것만**: 타입 매칭이 된 경우만 extern으로 인정
