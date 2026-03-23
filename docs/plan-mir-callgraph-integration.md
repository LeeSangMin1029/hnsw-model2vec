# MIR Call Graph Integration 계획

## 배경
- RA outgoing_calls: qdrant 750파일에서 OOM (salsa LRU 비활성)
- salsa 0.26 업그레이드: breaking changes로 불가
- salsa 0.25.2 LRU 패치: RA 로드 126초, 너무 느림
- **rustc MIR**: 87초(초기), 5.4초(증분), OOM 없음, 100% 정확 → 채택

## 실측 데이터
| 항목 | 우리 프로젝트 | qdrant |
|------|-------------|--------|
| cold | 60초 | 87초 |
| warm | 0.9초 | 1.1초 |
| 증분 | 1.2초 | 5.4초 |
| edges | 10,730 | 43,296 |
| OOM | 없음 | 없음 |

## 구현 범위

### 1. mir-callgraph 출력 정리
- caller_file: Debug format → 상대 경로 문자열
- JSON 출력 안정화
- 내부 함수만 필터 (std/외부 crate 호출 제외 옵션)

### 2. v-code add에서 MIR 통합
- `v-code add` 실행 시 `mir-callgraph`를 subprocess로 호출
- MIR edges를 graph.rs의 CallGraph에 주입
- 기존 이름 매칭(exact/short) 대신 MIR resolved edges 사용
- file_structure + 소스 파싱은 그대로 유지

### 3. 증분 파이프라인
- daemon의 file watcher가 변경 감지
- mir-callgraph 재실행 (변경 crate만 재컴파일)
- 변경 crate의 edges만 graph에서 교체

### 4. 구조
```
tools/mir-callgraph/           # nightly, 독립 workspace
  ├── rust-toolchain.toml
  ├── Cargo.toml
  └── src/main.rs              # RUSTC_WRAPPER + after_analysis

crates/v-code-intel/src/
  ├── graph.rs                 # CallGraph 구축 (MIR edges 사용)
  └── mir_edges.rs (신규)      # MIR edge 파싱 + subprocess 호출
```

## 검증
1. 우리 프로젝트: MIR graph vs 기존 이름 매칭 비교 → 동명 함수 disambiguation 확인
2. qdrant: 전체 graph 구축 → OOM 없음 확인
3. 증분: 파일 수정 → graph 자동 업데이트 → edge 정확성 확인
