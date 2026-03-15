# 프로젝트 기반 향후 작업 가능성

## 즉시 가능
- **IDE 플러그인**: daemon TCP JSON-RPC → VS Code/Zed extension (find, context, blast, jump)
- **PR 리뷰 자동화**: git diff → 변경 함수 → blast 영향 범위 → 리뷰 코멘트
- **코드 검색 MCP 서버**: Claude Code에서 v-code 직접 호출

## 중기
- **cross-repo 분석**: 여러 .v-code.db 연합 검색 (마이크로서비스 호출 추적)
- **리팩토링 추천 엔진**: dupes + blast radius → 병합 판단 자동화
- **코드 변경 이력 인덱싱**: git blame + call graph → 변경 빈도 분석

## 장기
- **자체 임베딩 파인튜닝**: chunk+call graph 학습 데이터 활용
- **아키텍처 시각화**: call graph 웹 UI (D3.js force graph)
- **다국어 통합 RAG**: 한국어 BM25 + 코드 검색 하나의 인터페이스
