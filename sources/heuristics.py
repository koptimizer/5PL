#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 4: heuristic policies against the exact two-stage robust LP (v14).

A policy = contract rule (first stage, no LP) x matching rule (second stage).
  contract rules : nominal   -> target = nominal demand               (Gamma_eff = 0)
                   bs_greedy -> target = nominal + Bertsimas-Sim(Gamma) buffer
                   full      -> target = nominal + full deviation      (Gamma_eff = J)
                   each filled cheapest (seller, carrier) pair first under supply and W/V caps
  matching rules : lp     -> exact LP recourse with x fixed (solve_lp fixed_x)
                   greedy -> surplus-ordered greedy per slot/scenario with FIFO backlog
Reference rows   : exact__lp (two-stage optimum) and exact__greedy (optimal x, greedy recourse).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from run_research import (
	RawData, aggregate, cfg_market, ext_params, load_or_create_raw, read_rows, run_key,
	sanitize_float, solve_lp, specialized_x_keys, theoretical_bs_buffer, trade_arcs, write_rows,
)

CONTRACT_RULES = ("nominal", "bs_greedy", "full")
MATCH_RULES = ("lp", "greedy")
XKey = Tuple[int, int, int, int]


def rule_gamma(rule: str, gamma: float, J: int) -> float:
	# Robustness budget each contract rule plugs into the Bertsimas-Sim buffer
	return {"nominal": 0.0, "bs_greedy": float(gamma), "full": float(J)}[rule]


def greedy_contract(cfg: Mapping[str, Any], raw: RawData, agg: Mapping[str, Any], gamma_eff: float, h_scale: float) -> Dict[XKey, float]:
	# Two passes per slot: (1) cheapest pair first within a carrier-share quota so both items fit the W/V mix,
	# (2) cheapest pair first for whatever target remains; supply and W/V caps always respected
	mkt = cfg_market(cfg)
	L, K = len(mkt["items"]), mkt["K"]
	pairs = sorted((float(h_scale) * raw.h[i, k, l], i, k, l) for (i, k, l, _) in specialized_x_keys(cfg, 1))
	x: Dict[XKey, float] = {}
	for t in range(agg["nslots"]):
		target = np.array([float(agg["d_hat"][:, l, t].sum()) + theoretical_bs_buffer(agg["xi"][:, l, t], gamma_eff) for l in range(L)])
		sup = agg["s_hat"][:, :, t].copy()
		cw = agg["cap_w"][:, t].copy()
		cv = agg["cap_v"][:, t].copy()
		share = agg["cap_w"][:, t] / float(agg["cap_w"][:, t].sum())
		quota = {(k, l): target[l] * share[k] for k in range(K) for l in range(L)}
		for pass_id in (0, 1):
			for _, i, k, l in pairs:
				item = mkt["items"][l]
				w, v = mkt["weight"][item], mkt["volume"][item]
				q = min(target[l], sup[i, l], cw[k] / w, cv[k] / v)
				if pass_id == 0:
					q = min(q, quota[k, l])
				if q <= 1e-9:
					continue
				x[i, k, l, t] = x.get((i, k, l, t), 0.0) + q
				quota[k, l] -= q
				target[l] -= q
				sup[i, l] -= q
				cw[k] -= w * q
				cv[k] -= v * q
	return x


def greedy_match(cfg: Mapping[str, Any], raw: RawData, agg: Mapping[str, Any], x: Mapping[XKey, float], h_scale: float, f_cost: float) -> Dict[str, Any]:
	# Surplus-ordered greedy recourse; negative-surplus arcs only close the to-date service gap and only while cheaper than rho
	mkt = cfg_market(cfg)
	ext = ext_params(cfg)
	scen = cfg["scenarios"]
	c_wait = float(cfg["economics"]["c_wait"])
	J, L, T = mkt["J"], len(mkt["items"]), agg["nslots"]
	order = sorted(trade_arcs(cfg), key=lambda a: -(raw.a[a[1], a[3]] - raw.b[a[0], a[3]] - raw.c[a[2], a[3]]))
	pre_cost = float(sum(float(h_scale) * raw.h[i, k, l] * q for (i, k, l, t), q in x.items()))
	auction_cost = float(f_cost) * T
	out: Dict[str, Any] = {"precontract_qty": float(sum(x.values())), "precontract_cost": pre_cost, "auction_cost": auction_cost, "n_auctions": T}
	obj = -pre_cost - auction_cost
	acc = {k: 0.0 for k in ["trade", "slot", "back", "term", "pen", "dem", "ship", "final"]}
	for s, sc in enumerate(scen):
		wgt, gsh = float(sc["weight"]), float(sc["gamma"])
		back = np.zeros((J, L))
		trade = slot_wait = back_wait = cum_d = cum_ship = 0.0
		for t in range(T):
			new_d = agg["d_real"][:, :, t, s]
			dem = new_d + back
			slot_wait += c_wait * float(agg["tau"][t]) * float(new_d.sum())
			if t > 0:
				back_wait += c_wait * float(agg["ell"][t]) * float(back.sum())
			cum_d += float(new_d.sum())
			rem = {key: q for key, q in x.items() if key[3] == t}
			cw = (1.0 - gsh) * agg["cap_w"][:, t].copy()
			cv = (1.0 - gsh) * agg["cap_v"][:, t].copy()
			shipped_t = 0.0
			for pass_id in (0, 1):
				for (i, j, k, l) in order:
					surplus = float(raw.a[j, l] - raw.b[i, l] - raw.c[k, l])
					if pass_id == 0 and surplus <= 0.0:
						break
					gap = ext["sl_min"] * cum_d - cum_ship - shipped_t
					if pass_id == 1 and (gap <= 1e-9 or surplus + ext["rho"] <= 0.0):
						break
					item = mkt["items"][l]
					w, v = mkt["weight"][item], mkt["volume"][item]
					cap = rem.get((i, k, l, t), 0.0)
					q = min(dem[j, l], cap, cw[k] / w, cv[k] / v)
					if pass_id == 1:
						q = min(q, gap)
					if q <= 1e-9:
						continue
					dem[j, l] -= q
					rem[i, k, l, t] = cap - q
					cw[k] -= w * q
					cv[k] -= v * q
					trade += surplus * q
					shipped_t += q
			cum_ship += shipped_t
			back = dem
		final = float(back.sum())
		term = c_wait * ext["kappa"] * float(agg["ell"][T - 1]) * final
		shortfall = max(0.0, ext["sl_min"] * cum_d - cum_ship)
		pen = ext["rho"] * shortfall
		out[f"service_{sc['name']}"] = cum_ship / cum_d if cum_d > 0 else float("nan")
		out[f"final_backlog_{sc['name']}"] = final
		out[f"service_shortfall_{sc['name']}"] = shortfall
		obj += wgt * (trade - slot_wait - back_wait - term - pen)
		for key, val in [("trade", trade), ("slot", slot_wait), ("back", back_wait), ("term", term), ("pen", pen), ("dem", cum_d), ("ship", cum_ship), ("final", final)]:
			acc[key] += wgt * val
	out.update({
		"objective": obj,
		"weighted_trade_surplus": acc["trade"],
		"weighted_slot_wait_cost": acc["slot"],
		"weighted_backlog_wait_cost": acc["back"],
		"weighted_terminal_backlog_cost": acc["term"],
		"weighted_service_penalty": acc["pen"],
		"weighted_service_rate": acc["ship"] / acc["dem"] if acc["dem"] > 0 else float("nan"),
		"weighted_final_backlog": acc["final"],
	})
	return out


REPORT_COLS = [
	"objective", "precontract_qty", "precontract_cost", "auction_cost", "weighted_trade_surplus",
	"weighted_slot_wait_cost", "weighted_backlog_wait_cost", "weighted_terminal_backlog_cost",
	"weighted_service_penalty", "weighted_service_rate", "weighted_final_backlog",
]


def method_row(base: Mapping[str, Any], method: str, res: Mapping[str, Any], exact_obj: float, runtime: float, scen_names: Sequence[str]) -> Dict[str, Any]:
	# One CSV row per (grid point, method) with gap to the exact two-stage optimum
	row = dict(base)
	row["method"] = method
	for c in REPORT_COLS:
		row[c] = res.get(c, float("nan"))
	for n in scen_names:
		row[f"service_{n}"] = res.get(f"service_{n}", float("nan"))
	row["gap_abs"] = exact_obj - float(res["objective"])
	row["gap_rel"] = row["gap_abs"] / abs(exact_obj) if exact_obj else float("nan")
	row["runtime_sec"] = runtime
	return row


def run_stage4(
	cfg: Mapping[str, Any],
	results_root: Path,
	seeds: Sequence[int],
	gammas: Sequence[float],
	f_costs: Sequence[float],
	deltas: Sequence[int],
	h_scale: float,
	resume: bool,
	solver_overrides: Optional[Mapping[str, Any]],
) -> None:
	# Grid loop: 4 LPs per point (exact + 3 contract rules with LP recourse) plus 4 greedy evaluations
	folder = results_root / "06_stage4_heuristics"
	folder.mkdir(parents=True, exist_ok=True)
	path = folder / "heuristic_summary.csv"
	if not resume and path.exists():
		path.unlink()
	done = set()
	for r in read_rows(path):
		done.add(run_key(int(r["seed"]), float(r["gamma"]), int(r["delta"]), float(r["h_scale"]), float(r["f_cost"])))
	mkt = cfg_market(cfg)
	scen_names = [sc["name"] for sc in cfg["scenarios"]]
	total = len(seeds) * len(gammas) * len(f_costs) * len(deltas)
	idx = 0
	for seed in seeds:
		raw = load_or_create_raw(cfg, results_root, int(seed))
		for gamma in gammas:
			for f_cost in f_costs:
				for delta in deltas:
					idx += 1
					key = run_key(seed, gamma, delta, h_scale, f_cost)
					tag = f"seed={seed}, Gamma={gamma}, F={f_cost}, Delta={delta}"
					if key in done:
						print(f"[stage4 {idx}/{total}] SKIP {tag}")
						continue
					print(f"[stage4 {idx}/{total}] RUN {tag}")
					agg = aggregate(raw, cfg, int(delta))
					base = {"seed": int(seed), "gamma": float(gamma), "f_cost": float(f_cost), "delta": int(delta), "h_scale": float(h_scale)}
					rows: List[Dict[str, Any]] = []
					sol: Dict[str, Any] = {}
					t0 = time.time()
					exact = solve_lp(cfg, raw, results_root, "stage4", int(seed), float(gamma), int(delta), float(h_scale),
						f_cost=float(f_cost), solution_out=sol, solver_overrides=solver_overrides)[0]
					exact_obj = float(exact["objective"])
					rows.append(method_row(base, "exact__lp", exact, exact_obj, time.time() - t0, scen_names))
					t0 = time.time()
					rows.append(method_row(base, "exact__greedy", greedy_match(cfg, raw, agg, sol["xR"], h_scale, f_cost), exact_obj, time.time() - t0, scen_names))
					for crule in CONTRACT_RULES:
						t0 = time.time()
						x = greedy_contract(cfg, raw, agg, rule_gamma(crule, gamma, mkt["J"]), h_scale)
						t_contract = time.time() - t0
						for mrule in MATCH_RULES:
							t0 = time.time()
							if mrule == "lp":
								res = solve_lp(cfg, raw, results_root, "stage4", int(seed), float(gamma), int(delta), float(h_scale),
									f_cost=float(f_cost), fixed_x=x, solver_overrides=solver_overrides)[0]
							else:
								res = greedy_match(cfg, raw, agg, x, h_scale, f_cost)
							rows.append(method_row(base, f"{crule}__{mrule}", res, exact_obj, t_contract + time.time() - t0, scen_names))
					write_rows(path, rows, append=True)
					best = min(rows[1:], key=lambda r: r["gap_rel"])
					print(f"    exact={exact_obj:.1f} best_heuristic={best['method']} gap={100 * best['gap_rel']:.2f}%")
	print(f"\n[stage4] complete. Results: {folder}")
	make_stage4_plots(path, folder, results_root / "figures" / "stage4")


def make_stage4_plots(path: Path, folder: Path, figdir: Path) -> None:
	# Gap-vs-Delta curves per F and the Delta* each method would recommend
	import pandas as pd
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	df = pd.read_csv(path)
	if df.empty:
		return
	figdir.mkdir(parents=True, exist_ok=True)
	keys = ["gamma", "f_cost"]
	mean_obj = df.groupby(keys + ["method", "delta"], as_index=False)[["objective", "gap_rel", "weighted_service_rate", "runtime_sec"]].mean()
	mean_obj.to_csv(folder / "heuristic_mean_by_delta.csv", index=False, encoding="utf-8-sig")
	for fc, sub in mean_obj.groupby("f_cost"):
		plt.figure(figsize=(9, 5))
		g = sub.groupby(["method", "delta"], as_index=False)["gap_rel"].mean()
		for method in sorted(g["method"].unique()):
			if method == "exact__lp":
				continue
			gg = g[g["method"] == method].sort_values("delta")
			plt.plot(gg["delta"], 100 * gg["gap_rel"], marker=".", label=method)
		plt.xlabel("Auction cycle Delta (days)")
		plt.ylabel("Mean relative gap to exact LP (%)")
		plt.title(f"Heuristic optimality gap vs Delta (F={fc:g}, mean over seeds and Gamma)")
		plt.legend(fontsize=8)
		plt.tight_layout()
		plt.savefig(figdir / f"heuristic_gap_vs_delta_F{sanitize_float(float(fc))}.png", dpi=180)
		plt.close()
	exact = mean_obj[mean_obj["method"] == "exact__lp"][keys + ["delta", "objective"]].rename(columns={"objective": "exact_mean_objective"})
	idx = mean_obj.groupby(keys + ["method"])["objective"].idxmax()
	star = mean_obj.loc[idx, keys + ["method", "delta", "objective"]].rename(columns={"delta": "delta_star_method", "objective": "method_mean_objective_at_star"})
	star = star.merge(exact, left_on=keys + ["delta_star_method"], right_on=keys + ["delta"], how="left").drop(columns=["delta"])
	best_exact = exact.loc[exact.groupby(keys)["exact_mean_objective"].idxmax(), keys + ["delta", "exact_mean_objective"]].rename(columns={"delta": "delta_star_exact", "exact_mean_objective": "exact_at_exact_star"})
	star = star.merge(best_exact, on=keys, how="left")
	star["delta_choice_regret"] = star["exact_at_exact_star"] - star["exact_mean_objective"]
	star.to_csv(folder / "delta_star_by_method.csv", index=False, encoding="utf-8-sig")
	summary = df[df["method"] != "exact__lp"].groupby(["method"], as_index=False).agg(
		mean_gap_rel=("gap_rel", "mean"), max_gap_rel=("gap_rel", "max"), mean_runtime=("runtime_sec", "mean"),
		mean_service=("weighted_service_rate", "mean"))
	summary.to_csv(folder / "heuristic_overall_summary.csv", index=False, encoding="utf-8-sig")
	print(summary.to_string(index=False))
