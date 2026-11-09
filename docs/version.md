# 버전 히스토리

버전은 "모형 + 실험 사이클" 단위로 매긴다. 각 항목은 변경 내용, 실험 그리드, 결과 위치, 핵심 결론을 적는다.

## v13 — 통합모형 9.2 Baseline LP (기준점, 2026-09-02 실행 / 2026-11-09 정리)
- 모형: `docs/tex/중견연구 통합모형_9.2.tex`의 Baseline. 연속변수 2-stage robust LP
  (Bertsimas–Sim 선제계약 `xR, eta, mu`, N:M 실행매칭 `y`, 이월수요 `m`). 경매운영비, minimum lot,
  Big-M, 서비스율 제약 없음. 가격은 MMTR–MVCG 사후처리로 분리(미구현).
- 코드: `sources/run_research.py`, 설정 `sources/config_baseline.json` (results_subdir = v13)
- 실험: Stage 0 (12 LP), Stage 1 (5 seed × Γ{0,2,4,6,8} × Δ1..28 = 700 LP), Stage 2A (60 LP), Stage 3 regret.
  Stage 2B는 2026-11-09 사이클에서 뒤늦게 실행 (`scripts/run_stage2b_v13.sh`).
- 결과: `sources/results/v13/`
- 결론: 모든 seed·Γ에서 Δ* = 1, regret 0. Γ 0→8에서 서비스율 0.969→0.990, 목적값 −3.4%.
  Stage 2A에서 h^R 배율은 선제계약량을 ±5% 이내로만 움직이고 Γ=8에서는 전혀 움직이지 않음(robust cover가 지배).
  긴 주기의 이익(pooling)이 모형에 없으므로 Δ=1 우세는 구조적 결과.

## v14 — 경매운영비 + 종료 backlog 패널티 + soft 서비스율 + heuristic 비교 + MMTR–MVCG 가격 (2026-11-09 시작)
- 모형 확장 (`config_v14.json`의 `extensions` 절; 키가 없으면 v13과 비트 단위로 동일한 LP):
  1. 경매 1회 고정비 F: 목적함수에 −F·|T(Δ)|. F ∈ {100, 200, 400} outer grid (h_scale과 같은 차원).
  2. 종료 backlog 패널티: 마지막 슬롯 잔여 backlog에 c_wait·κ·ℓ_T 부과 (κ = 1.0).
  3. Soft 서비스율: 시나리오별 u_s ≥ 0.95·D_s − shipped_s, 패널티 ρ = 100/unit. (적합성 검토 3번 지적 반영: 평균이 아닌 시나리오별)
- 코드 변경:
  - `run_research.py`: `f_cost` grid 차원, `fixed_x`(주어진 선제계약을 정확한 LP recourse로 평가), `solution_out`,
    Stage 3를 (Γ, F) 그룹으로 일반화 + 목적함수 분해 csv/그림, 결과 루트 `results/<results_subdir>`.
  - `heuristics.py` (Stage 4): 계약규칙 {nominal, bs_greedy, full} × 매칭규칙 {lp, greedy} + exact__greedy. 지표: 정확해 대비 gap, 서비스율, 각 방법이 고르는 Δ*와 그로 인한 regret.
  - `pricing.py`: Guo et al. (2022, TR-B) MMTR–MVCG 8단계를 (item, slot, scenario) 시장에 적용. 구매자 bid a, (판매자, 운송업체) 묶음 bid b+c, 묶음 용량 = xR, 첫 배분 = LP의 y. 플랫폼 마진, IR 위반, 예산균형, 효율 보고.
- 실험 그리드: Stage 0 (12), Stage 1 (5 seed × Γ 5 × F 3 × Δ 28 = 2100 LP), Stage 3, Stage 4 (5 seed × Γ{0,4,8} × F 3 × Δ 28 = 1260점 × 4 LP), pricing (seed 101, Γ{0,4,8} × F 3의 Δ*).
  연쇄 스크립트 `scripts/run_v14_chain.sh`.
- 결과: `sources/results/v14/`
- 검증: config_baseline으로 `one --seed 101 --gamma 4 --delta 7` 실행 시 v13 Stage 2A 값 801999.337과 일치 (확장이 기본값에서 비활성임을 확인).
- 결론: (실험 종료 후 기록)
