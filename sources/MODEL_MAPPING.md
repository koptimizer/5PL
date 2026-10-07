# 통합모형 9.2 ↔ Gurobi 코드 매핑

| 논문 구성 | 코드 구현 |
|---|---|
| 일별 명목수요 `d_hat ~ Poisson(10)` | `generate_raw_data()` |
| 공급여유 `rhoS=1.25` | demand-linked item total supply |
| 운송용량여유 `rhoC=1.25` | demand-linked total W/V capacity |
| 최대수요편차 `xi=0.20*d_hat` | raw `xi` array |
| Gamma `{0,2,4,6,8}` | robust cover outer sensitivity |
| `A_t(Delta)` | `aggregate()` day groups |
| `tau=(ell-1)/2` | `aggregate()` |
| Stage-1 `xR, eta, mu` | `solve_lp()` |
| Robust cover | `robust_cover`, `robust_dual` constraints |
| 판매자 공급상한 | `pre_supply` |
| Stage-1 W/V capacity | `pre_weight`, `pre_volume` |
| Stage-2 demand balance | `demand_balance` |
| 계약사용한도 | `contract_use` |
| Stage-2 shock capacity | `run_weight`, `run_volume` |
| 서비스율 | post-solve KPI |
| Delta outer-loop | `run_grid()` |
| 공통 추천 Delta | `stage3_regret()` |
| Regret | `objective_seed_opt - objective_recommended` |

## 의도적으로 구현하지 않은 것

- `z`, minimum lot, Big-M: Baseline에서 삭제됨.
- 경매 1회 운영비용: Baseline에서 삭제됨.
- 서비스율 hard constraint: KPI로만 보고.
- 지역/거리/노선 선호: 없음.
- seller-buyer 사전 pair: 없음.
- MMTR-MVCG 가격식: 최신 통합문서가 구체적인 실행 알고리즘을 제공하지 않으므로 임의 구현하지 않음.
