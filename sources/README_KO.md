# 5PL Gurobi 연구 파이프라인 — 통합모형 9.2 Baseline (v13) + v14 확장

이 패키지는 `../docs/tex/중견연구 통합모형_9.2.tex`의 **Baseline two-stage robust LP**를 연구 순서대로 실행하도록 만든 코드입니다.
버전별 변경 내용은 `../docs/version.md`를 참고하세요.

## v14 요약 (2026-11)

- 설정 파일로 버전을 고릅니다: `--config config_baseline.json` (v13, 결과 `results/v13/`), `--config config_v14.json` (v14, 결과 `results/v14/`).
- `extensions` 절이 없으면 v13 LP와 완전히 같은 결과가 나옵니다 (검증: seed 101, Γ=4, Δ=7 → 801999.337 일치).
- v14 확장: 경매 1회 고정비 `F` (outer grid), 종료 backlog 패널티 `κ`, 시나리오별 soft 서비스율 (`SL_min`, `ρ`).
- 새 명령:
  - `stage4`: `heuristics.py` — 계약규칙 {nominal, bs_greedy, full} × 매칭규칙 {lp, greedy} + exact__greedy를 정확해와 비교. 결과 `06_stage4_heuristics/`, 그림 `figures/stage4/`.
  - `pricing`: `pricing.py` — Guo et al. (2022) MMTR–MVCG 가격청산을 Stage 3의 Δ*에 적용. 결과 `07_pricing/`, 그림 `figures/pricing/`.
  - `stage3`는 (Γ, F) 그룹별 추천 Δ*, regret, 목적함수 분해(`objective_decomposition.csv`)를 냅니다.
- 전체 연쇄 실행: `bash scripts/run_v14_chain.sh` (check → stage0 → stage1 → stage3 → stage4 → pricing, 로그 `results/v14/logs/`).

---

## 0. 무엇을 구현했나

Core optimization은 모두 연속변수입니다.

- 1단계: robust 선제계약 `xR`, Bertsimas–Sim 보조변수 `eta`, `mu`
- 2단계: N:M 실제 운송량 `y`, 이월수요 `m`
- 실행 시나리오: No-shock / Demand Surge / Network Disruption
- `Delta`는 LP 내부 변수가 아니라 outer-loop에서 `1..28`을 완전열거
- 같은 seed에서는 Gamma, Delta, hR scale이 달라도 raw data를 다시 뽑지 않음
- `z`, 최소거래량, Big-M 없음
- MMTR–MVCG 가격은 임의로 구현하지 않음. 필요하면 `positive_trades.csv`를 가격청산 모듈 입력으로 사용

## 1. 설치

Python 3.10+ 권장.

```bash
python -m pip install -r requirements.txt
```

Gurobi license가 정상인지 확인:

```bash
python -c "import gurobipy as gp; print(gp.gurobi.version())"
```

## 2. 연구 순서대로 실행

### Step A — 환경/데이터 확인 (Gurobi 없이도 raw data 생성 가능)

```bash
python run_research.py check
```

확인 내용:

- Seed 101 raw data 생성 및 SHA256 저장
- 평균 일일수요/공급비율
- 명목 중량·부피 utilization
- Gamma 0/4/8, Delta 1/7/14/28에 대한 구조적 필요조건 점검
- Gurobi import/license 초기화 여부

### Stage 0 — Sanity check: 12 LPs

```bash
python run_research.py stage0
```

조합:

- seed = 101
- Gamma = {0, 4, 8}
- Delta = {1, 7, 14, 28}
- h-scale = 1

반드시 `results/01_stage0_sanity/sanity_report.txt`를 먼저 확인합니다.

### Stage 1 — Baseline 본실험: 700 LPs

Stage 0가 정상일 때만 실행합니다.

```bash
python run_research.py stage1
```

조합:

- seeds = {101, 211, 307, 401, 503}
- Gamma = {0, 2, 4, 6, 8}
- Delta = 1, ..., 28
- 총 700 LPs

중간에 종료되어도 같은 명령을 다시 실행하면 완료된 조합은 자동으로 건너뜁니다.

### Stage 2A — 선제계약비용 sensitivity pilot: 60 LPs

```bash
python run_research.py stage2a
```

조합:

- seed = 101
- Gamma = {0, 4, 8}
- lambda_h = {0.5, 0.75, 1.0, 1.5, 2.0}
- Delta = {1, 7, 14, 28}

먼저 이 pilot으로 `hR` 배율이 실제로 선제계약과 목적값을 움직이는지 봅니다.

### Stage 2B — 필요할 때만 hR sensitivity 전체 Delta: 420 LPs

```bash
python run_research.py stage2b
```

조합:

- seed = 101
- Gamma = {0, 4, 8}
- lambda_h = {0.5, 0.75, 1.0, 1.5, 2.0}
- Delta = 1, ..., 28

Stage 2A에서 의미 있는 변화가 확인될 때 수행하는 것을 권장합니다.

### Stage 3 — 공통 추천 경매주기 및 Regret

Gurobi를 다시 풀지 않고 Stage 1 결과를 이용합니다.

```bash
python run_research.py stage3
```

생성 결과:

- `per_seed_optima.csv`
- `recommended_delta_by_gamma.csv`
- `regret_by_seed_gamma.csv`
- `regret_summary_by_gamma.csv`

## 3. 한 번에 실행

`stage2b`는 제외하고 Core 순서만 실행합니다.

```bash
python run_research.py all
```

순서:

`check -> stage0 -> stage1 -> stage2a -> stage3`

700 LPs가 포함되므로 처음에는 각 단계별로 실행하는 것을 권장합니다.

## 4. 특정 조합 하나만 테스트

예: seed 101, Gamma 4, Delta 7

```bash
python run_research.py one --seed 101 --gamma 4 --delta 7
```

MMTR–MVCG 후처리용으로 실제 양의 운송량 거래까지 저장하려면:

```bash
python run_research.py one --seed 101 --gamma 4 --delta 7 --export-trades
```

## 5. 원하는 grid만 직접 실행

예:

```bash
python run_research.py custom-grid --seeds 101,211 --gammas 0,4,8 --deltas 1-14 --h-scales 1.0
```

## 6. 결과 폴더

```text
results/
  00_raw_data/
  01_stage0_sanity/
  02_stage1_baseline/
  03_stage2_hcost_pilot/
  04_stage2_hcost_full/
  05_stage3_regret/
  figures/
  logs/
  iis/
```

각 최적화 단계에는 기본적으로 다음 CSV가 생깁니다.

- `run_summary.csv`
- `robust_diagnostics.csv`
- `scenario_summary.csv`
- `capacity_diagnostics.csv`

### run_summary 핵심 열

- `objective`: 순운영잉여
- `precontract_qty`: 전체 선제계약량
- `precontract_cost`
- `weighted_trade_surplus`
- `weighted_slot_wait_cost`
- `weighted_backlog_wait_cost`
- `weighted_service_rate`
- `weighted_final_backlog`
- `runtime_sec`
- `simplex_iterations`, `barrier_iterations`
- W/V capacity binding rate

LP이므로 **MIP gap은 기록하지 않습니다.**

### robust_diagnostics 핵심 열

- `theoretical_target`: Bertsimas–Sim 이론적 robust target
- `model_robust_rhs`: eta/mu가 만든 실제 robust RHS
- `precontract_qty`
- `slack_vs_theory`
- `slack_vs_model_rhs`

`theoretical_target`과 `model_robust_rhs`가 맞는지 확인할 수 있습니다.

## 7. 현재 목적함수의 대기비용 구현

문서 9.2를 그대로 따라 다음처럼 구현했습니다.

1. 슬롯에 새로 유입된 전체 실현수요는 평균 매칭대기시간 `tau`를 부담합니다.
2. 이전 슬롯에서 넘어온 backlog는 현재 슬롯 길이 `ell`만큼 추가 대기비용을 부담합니다.
3. 마지막 슬롯 말에 남은 backlog에는 horizon 이후의 추가 대기비용을 임의로 붙이지 않습니다. 대신 `final_backlog` KPI로 저장합니다.

이 마지막 항목은 **문서 수식을 그대로 따른 것**이며, 필요하면 추후 terminal backlog penalty 확장으로 별도 검토할 수 있습니다.

## 8. 중요한 Sanity 해석

### Gamma target은 증가하는데 xR가 그대로다

다른 supply/capacity 제약이 지배하거나 이미 충분히 계약한 경우일 수 있습니다. `robust_diagnostics.csv`와 `capacity_diagnostics.csv`를 같이 봅니다.

### Network Disruption과 No-shock 차이가 거의 없다

현재 `rhoC=1.25`, `gamma=0.15`이므로 aggregate 관점에서는 충격 후에도 약 `1.0625 x nominal demand`의 물리용량이 남을 수 있습니다. 이는 코드 오류가 아니라 calibration 결과일 수 있습니다.

### Delta=1이 항상 최적이다

현재 Baseline에는 경매 1회 고정비용과 minimum lot이 없습니다. 긴 주기의 이익을 억지로 만들지 않았으므로 `Delta=1`이 우세할 수 있습니다. 먼저 이 결과를 Baseline 결과로 받아들인 뒤, 필요하면 확장모형에서 discrete pre-contracting 또는 다른 실제 운영요인을 검토하는 것이 안전합니다.

## 9. 재현성

동일 seed의 raw data는 `results/00_raw_data/seed_XXX.npz`에 캐시됩니다. 같은 seed에서는 Gamma, Delta, h-scale을 바꿔도 이 파일을 재사용합니다.

config가 바뀌었는데 예전 raw data가 남아 있으면 실행을 중단하도록 만들었습니다. 의도적으로 새 데이터를 만들려면:

```bash
python run_research.py --overwrite-data check
```

주의: 이 옵션은 이후 비교의 기준 raw data를 바꿀 수 있으므로 실험 중간에는 사용하지 않는 것이 좋습니다.

## 10. Solver 옵션

예:

```bash
python run_research.py --threads 8 --time-limit 1200 stage1
```

Gurobi 로그를 화면에 보이게 하려면:

```bash
python run_research.py --solver-output stage0
```

개별 로그파일도 남기려면:

```bash
python run_research.py --write-logs stage0
```

## 11. 문서와 코드 매핑

- `xR` → Stage-1 robust pre-contract capacity
- `eta`, `mu` → budgeted uncertainty robust counterpart
- `y` → scenario-dependent N:M execution shipment
- `m` → backlog carried to next auction slot
- `Gamma` → robustness budget
- `Delta` → external auction-cycle candidate
- `tau` → within-slot average matching wait
- `ell` → actual slot length

`z`, minimum-lot, Big-M은 Model 9.2 Baseline에서 사용하지 않습니다.
