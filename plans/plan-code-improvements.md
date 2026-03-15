# 코드 개선 계획: 중복 제거 + 테스트 + 순환 의존 해소

## 1. 프로덕션 코드 중복 제거

### A. install_handler 통합
- v-daemon/interrupt.rs, v-hnsw-cli/interrupt.rs → v-hnsw-core에 한 번 정의

### B. CallGraph::build 내부 루프 추출
- build()와 build_with_resolved_calls()의 공통 루프를 내부 함수로 추출
- resolve_chunk_edges() 같은 이름으로 owner_types/imports/enriched_types/resolve 로직 통합

### C. callee 추출 로직 통합
- walk_for_calls_with_lines L87-105 ↔ walk_for_string_args_inner L239-257
- extract_callee_from_node(node, src) 공통 함수 추출

### D. StorageEngine insert 패턴
- insert()와 replace_source()의 vector+payload 버퍼링 공통화

## 2. v-code-intel 테스트 추가
- 179 prod 함수, 0 직접 테스트 → 핵심 함수 단위 테스트 추가
- resolve_with_imports, context_cmd 등

## 3. 순환 의존 해소 (별도 크레이트 없이)
- daemon client 함수(port_path, read_port, is_running, daemon_rpc, notify_reload)를 v-hnsw-storage로 이동
- v-hnsw-cli → v-daemon 의존 제거, v-hnsw-storage의 daemon_client 모듈 사용
- v-daemon → v-hnsw-cli + v-code-chunk 의존 추가 가능 (순환 해소)
- auto_reindex: 서브프로세스 → 라이브러리 직접 호출 (모델 재사용)
