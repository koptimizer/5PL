# 버전 히스토리

버전은 "모형 + 실험 사이클" 단위로 매긴다. 각 항목은 변경 내용, 실험 그리드, 결과 위치, 핵심 결론을 적는다.

## v13 — 통합모형 9.2 Baseline LP (기준점, 2026-09-02 실행 / 2026-11-09 정리)
- 모형: `docs/tex/중견연구 통합모형_9.2.tex`의 Baseline. 연속변수 2-stage robust LP
  (Bertsimas–Sim 선제계약 `xR, eta, mu`, N:M 실행매칭 `y`, 이월수요 `m`). 경매운영비, minimum lot,
  Big-M, 서비스율 제약 없음. 가격은 MMTR–MVCG 사후처리로 분리(미구현).
- 코드: `sources/run_research.py`, 설정 `sources/config_baseline.json` (results_subdir = v13)
- 실험: Stage 0 (12 LP), Stage 1 (5 seed × Γ{0,2,4,6,8} × Δ1..28 = 700 LP), Stage 2A (60 LP), Stage 3 regret.
  Stage 2B는 2026-11-09 사이클에서 뒤늦게 실행 (`scripts/run_stage2b_v13.sh`, 420 LP 모두 OPTIMAL, 약 1분).
  결과: h_scale 0.5~2.0 전 구간, Γ{0,4,8} 모두에서 Δ* = 1 유지. 선제계약량은 Γ=0에서 24769→22405 (−9.5%),
  Γ=4에서 25490→25113 (−1.5%), Γ=8에서는 26852로 고정. 즉 h^R는 Δ* 선택에 영향이 없고, Γ가 클수록 robust cover가 계약량을 지배.
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
- Stage 1·3 결과 (2026-11-09 18:12, 2100 LP 전부 OPTIMAL, 약 9분):
  - 처음으로 내부 최적 경매주기가 나옴: F=100, 200 → Δ* = 2, F=400 → Δ* = 3. **모든 Γ(0~8)와 모든 seed에서 동일** (seed별 Δ* 표준편차 0, regret 0).
  - 원인 (Γ=4, F=200 분해): Δ 1→2에서 경매운영비 −14000, 선제계약비 −1258 vs 슬롯 대기비 +5766, backlog 대기 +539 → 순이익 +8803.
    Δ 2→3에서는 경매운영비 절감이 −4600으로 줄어 대기비 증가 +5681을 못 넘음 → −742.
  - 닫힌 형태로 설명 가능: 슬롯 대기비는 Δ마다 거의 일정하게 c_wait·D/2 ≈ 5600씩 증가하고, 경매운영비 절감은 F·140/(Δ(Δ−1)).
    따라서 Δ* = max{Δ : Δ(Δ−1) < F/40}. F=100 → 2, F=200 → 2, F=400 → 3로 실험과 정확히 일치.
  - 거래잉여와 선제계약비는 Δ에 거의 무관(각각 ±0.02%, −0.4%/step). 즉 현재 Baseline 데이터에서는 "pooling 이익"이 사실상 없고,
    Δ*는 F와 c_wait의 비율로만 결정되며 Γ와 독립. 실현수요가 d_real = (1+θ)·d_hat의 결정적 스케일이라 집계에 의한 분산감소 효과가 모형에 없는 것이 구조적 이유.
  - Soft 서비스율 패널티는 모든 (Γ, F, Δ)에서 0 (LP가 자발적으로 95%를 넘김). 서비스 하한은 Baseline에서는 비구속.
- Stage 4 결과 (18:30, 1260점 × 4 LP, 약 18분). 정확해 대비 평균 상대 gap (Γ{0,4,8}·F·Δ 평균):
  | 정책 | 평균 gap | 최대 gap | 서비스율 |
  |---|---|---|---|
  | full__lp (전량 커버 계약 + LP 매칭) | 7.3% | 14.6% | 0.998 |
  | bs_greedy__lp (Γ-buffer 계약 + LP 매칭) | 11.2% | 33.7% | 0.985 |
  | exact__greedy (최적 계약 + greedy 매칭) | 17.8% | 33.7% | 0.945 |
  | nominal__lp | 18.5% | 33.7% | 0.968 |
  | bs_greedy__greedy / nominal__greedy / full__greedy | 18.9 / 21.2 / 24.1% | 44–47% | 0.95 |
  - 해석: (1) 2단계 매칭을 greedy로 하면 계약이 최적이어도 17.8%를 잃음 → 손실의 대부분은 매칭 단계. 최적 x는 LP 매칭에 맞춰 빡빡하게 짜여 있어 greedy 하에서는 오히려 취약(서비스 0.945).
    (2) 계약 heuristic 중에서는 "전량 커버(Γ=J)"가 가장 좋음: 과잉계약 비용(+ ~7%)이 서비스 패널티 ρ=100보다 싸기 때문. bs_greedy는 Γ=8에서 full과 동일(5.0%), Γ=0에서는 nominal과 동일(19.9%).
    (3) Δ* 선택은 heuristic으로도 거의 재현됨: 9개 (Γ,F) 조합 중 full/nominal 계열은 모두 정확해와 같은 Δ*, bs_greedy·exact__greedy는 3~4개 조합에서 ±1 어긋나며 그로 인한 regret ≤ 1803 (< 0.3%).
    (4) 실행시간: greedy 0.014s, heuristic+LP recourse 0.17s, 정확해 LP 0.03~1.4s. 이 규모에서는 정확해가 충분히 빠름.
- Pricing 결과 (18:31, seed 101, Γ{0,4,8} × F 3의 Δ*, 총 3366개 (item, slot, scenario) 시장, 각 3~6초):
  - 27개 (Γ,F,시나리오) 조합 모두 개별합리성 위반 0, 예산균형(플랫폼 마진 ≥ 0) 충족 → Guo et al. 정리 3의 성질이 적용판에서도 유지.
  - Trade reduction으로 거래량의 13~19%가 제거됨(수량 효율 0.81~0.87), 후생 효율은 0.96~0.98 (한계거래만 빠지므로 후생 손실은 작음).
    Γ=4·Demand_Surge에서 1.05 > 1인 것은 LP의 y가 서비스 패널티 때문에 음(−)의 잉여 arc까지 실었기 때문(시장 LP는 정적이라 이를 싣지 않음).
  - 구매자 청산가 평균 128(No-shock)~139(Surge), 판매자+운송업체 지급 115~119 → 플랫폼 마진율 9.5%(No-shock)~14.7%(Surge/Disruption). 충격 시나리오에서 한계거래가 비싸지며 마진이 커지는 방향.
  - 가격은 Δ* 선택에 영향을 주지 않음(사후처리). 결과: `07_pricing/`, 그림 `figures/pricing/`.
- 사이클 종료 (2026-11-13): 대용량 진단 csv(capacity/robust_diagnostics, positive_trades, pricing 시장/에이전트 csv)는 git에서 제외하고
  히스토리를 재작성 (pack 58MB). push는 GitHub 인증 부재(403)로 보류.

## v15 — 일 단위 시장 공통 수요 충격으로 pooling 이익 모형화 (2026-11-13 시작)
- 동기: v14에서 Δ*가 Γ와 완전히 독립이었던 구조적 이유(실현수요가 결정적 스케일)를 해소.
- 데이터 변경 (`config_v15.json`, `data.realization_noise_rate = 0.2`, `realization_noise_mode = "market"`):
  - 품목·일별 공통 계수 ε_{l,r} ~ U[0.8, 1.2]를 뽑아 실현 일수요 = 명목 × ε. 모든 실행 시나리오가 같은 ε를 공유하고 θ로 스케일.
  - 슬롯 집계 시 일별 편차가 평균화되어 |실현−명목|/명목이 Δ=1: 6.7% → Δ=7: 2.5% → Δ=28: 1.1% (seed 101).
  - |ε−1| ≤ 0.2 = 수요편차율이므로 실현값은 항상 budgeted uncertainty set 내부 (Γ=J 커버가 최악치를 보장).
  - 명목 데이터(수요·공급·용량·경제 파라미터)는 RNG 하위스트림을 뒤에 추가해 v13/v14와 비트 단위 동일. v14 설정으로 돌리면 결과 불변 (796713.672 재확인).
  - "buyer" 모드(구매자별 독립 노이즈)도 구현했으나 집계 제약이 구매자 합이라 Δ=1에서 이미 평균화됨(편차 2.3%) → 주 실험은 "market" 모드.
- 코드: `RawData.eps`, `realized_daily_demand()`, `aggregate()`의 d_real이 실현 일수요 합을 사용. heuristics/pricing은 agg를 쓰므로 자동 반영.
- 실험 그리드: v14와 동일 (Stage 0/1/3/4/pricing, F{100,200,400}), 연쇄 스크립트 `scripts/run_chain.sh config_v15.json`.
- 사전 관찰 (seed 101, F=200): Γ=0에서 backlog 대기비 Δ1 9024 → Δ7 7894, Γ=8에서 1459 → 411. pooling 이익은 존재하나 슬롯 대기비 증가(≈5600/step)보다 작음.
- 결과: `sources/results/v15/` (실험 종료 후 기록)
