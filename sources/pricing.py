#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MMTR-MVCG post-processing pricing, adapted from Guo, Zhang, Cheng & Zhao (2022),
"Truthful double auction mechanisms for online freight platforms with transaction costs",
Transportation Research Part B 158, Section 4.2 (Steps 1-8), to the 5PL three-sided market.

Adaptation (one market per item l, slot t, scenario s):
  buyers  j        bid a_jl, quantity D = realised demand + carried backlog
  bundles (i, k)   bid b_il + c_kl, quantity = pre-contract xR[i,k,l,t]  (seller and carrier sold as one unit)
  transaction cost d = 0 (no lane cost in the Baseline), so Step 3 removes whole buyers with a_j <= p+
  carrier W/V capacity for item l = (1-gamma_s) cap minus the LP allocation of the other item
  first allocation q0 = the two-stage LP's y (the matching the platform actually executes)
Payments follow (15)/(16): bundle bids replaced by p+ for buyer payments, buyer bids replaced by p-
for bundle revenues, each computed as a VCG marginal contribution on the remaining agents.
Bundle revenue is split seller : carrier in proportion b_il : c_kl (reporting assumption).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from run_research import (
	RawData, aggregate, cfg_market, import_gurobi, load_or_create_raw, read_rows, sanitize_float,
	solve_lp, write_rows,
)

Buyer = Tuple[int, float, float]            # (j, bid, quantity)
Bundle = Tuple[Tuple[int, int], float, float]  # ((i, k), bid, capacity)


def market_lp(gp, GRB, buyers: Sequence[Buyer], bundles: Sequence[Bundle], cap_w: Mapping[int, float], cap_v: Mapping[int, float],
	w: float, v: float, buyer_bid: Optional[float] = None, bundle_bid: Optional[float] = None,
	skip_buyer: Optional[int] = None, skip_bundle: Optional[Tuple[int, int]] = None) -> Tuple[float, Dict[Tuple[int, int, int], float]]:
	# Efficient allocation value V and quantities q for one market, with optional bid replacement / agent removal
	m = gp.Model("mkt")
	m.Params.OutputFlag = 0
	m.Params.Threads = 1
	q = {}
	for j, a, D in buyers:
		if j == skip_buyer or D <= 1e-9:
			continue
		for (i, k), cost, cap in bundles:
			if (i, k) == skip_bundle or cap <= 1e-9:
				continue
			val = (a if buyer_bid is None else buyer_bid) - (cost if bundle_bid is None else bundle_bid)
			q[j, i, k] = m.addVar(lb=0.0, obj=val)
	if not q:
		return 0.0, {}
	m.ModelSense = GRB.MAXIMIZE
	for j, a, D in buyers:
		m.addConstr(gp.quicksum(var for (jj, i, k), var in q.items() if jj == j) <= D)
	for (i, k), cost, cap in bundles:
		m.addConstr(gp.quicksum(var for (j, ii, kk), var in q.items() if (ii, kk) == (i, k)) <= cap)
	for k in cap_w:
		m.addConstr(gp.quicksum(w * var for (j, i, kk), var in q.items() if kk == k) <= cap_w[k])
		m.addConstr(gp.quicksum(v * var for (j, i, kk), var in q.items() if kk == k) <= cap_v[k])
	m.optimize()
	return float(m.ObjVal), {key: float(var.X) for key, var in q.items() if var.X > 1e-9}


def price_market(gp, GRB, raw: RawData, mkt: Mapping[str, Any], l: int, buyers: Sequence[Buyer], bundles: Sequence[Bundle],
	cap_w: Mapping[int, float], cap_v: Mapping[int, float], y0: Mapping[Tuple[int, int, int], float]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
	# Steps 2-8 of MMTR-MVCG for one market; returns a market row and per-agent payment rows
	item = mkt["items"][l]
	w, v = mkt["weight"][item], mkt["volume"][item]
	bid_j = {j: a for j, a, _ in buyers}
	bid_b = {ik: c for ik, c, _ in bundles}
	q0_j: Dict[int, float] = {}
	q0_b: Dict[Tuple[int, int], float] = {}
	for (j, i, k), q in y0.items():
		q0_j[j] = q0_j.get(j, 0.0) + q
		q0_b[i, k] = q0_b.get((i, k), 0.0) + q
	row: Dict[str, Any] = {"item": item, "qty_lp": float(sum(y0.values())), "welfare_lp": float(sum((bid_j[j] - bid_b[i, k]) * q for (j, i, k), q in y0.items()))}
	if not y0:
		row.update({"p_plus": float("nan"), "p_minus": float("nan"), "qty_mech": 0.0, "welfare_mech": 0.0, "n_buyers_trading": 0, "n_bundles_trading": 0,
			"buyer_payment_total": 0.0, "bundle_revenue_total": 0.0, "seller_revenue_total": 0.0, "carrier_revenue_total": 0.0,
			"platform_margin": 0.0, "buyer_unit_price_avg": float("nan"), "bundle_unit_price_avg": float("nan"), "ir_violations": 0, "n_small_lps": 0})
		return row, []
	# Step 2: threshold prices from the first allocation
	low_buyer = min(bid_j[j] for j in q0_j)
	high_bundle = max(bid_b[ik] for ik in q0_b)
	p_plus = max(low_buyer, high_bundle)
	p_minus = min(low_buyer, high_bundle)
	# Steps 3-4: remaining agents
	r_buyers = [b for b in buyers if b[1] > p_plus and b[2] > 1e-9]
	r_bundles = [b for b in bundles if b[1] < p_minus and b[2] > 1e-9]
	n_lp = 1
	V, q = market_lp(gp, GRB, r_buyers, r_bundles, cap_w, cap_v, w, v)
	# Step 5: trading sets
	Q_j: Dict[int, float] = {}
	Q_b: Dict[Tuple[int, int], float] = {}
	for (j, i, k), val in q.items():
		Q_j[j] = Q_j.get(j, 0.0) + val
		Q_b[i, k] = Q_b.get((i, k), 0.0) + val
	agents: List[Dict[str, Any]] = []
	pay_total = rev_total = seller_total = carrier_total = 0.0
	ir_bad = 0
	# Step 6: buyer payments with bundle bids replaced by p+
	if Q_j:
		V_plus, _ = market_lp(gp, GRB, r_buyers, r_bundles, cap_w, cap_v, w, v, bundle_bid=p_plus)
		n_lp += 1
		for j, Qj in Q_j.items():
			V_minus_j, _ = market_lp(gp, GRB, r_buyers, r_bundles, cap_w, cap_v, w, v, bundle_bid=p_plus, skip_buyer=j)
			n_lp += 1
			pay = bid_j[j] * Qj - (V_plus - V_minus_j)
			ir_bad += int(pay > bid_j[j] * Qj + 1e-6)
			pay_total += pay
			agents.append({"item": item, "agent": "buyer", "id": j + 1, "bid": bid_j[j], "quantity": Qj, "payment": pay, "unit_price": pay / Qj})
	# Step 7: bundle revenues with buyer bids replaced by p-
	if Q_b:
		V_minus, _ = market_lp(gp, GRB, r_buyers, r_bundles, cap_w, cap_v, w, v, buyer_bid=p_minus)
		n_lp += 1
		for (i, k), Qb in Q_b.items():
			V_minus_b, _ = market_lp(gp, GRB, r_buyers, r_bundles, cap_w, cap_v, w, v, buyer_bid=p_minus, skip_bundle=(i, k))
			n_lp += 1
			rev = bid_b[i, k] * Qb + (V_minus - V_minus_b)
			ir_bad += int(rev < bid_b[i, k] * Qb - 1e-6)
			rev_total += rev
			share_s = float(raw.b[i, l]) / float(raw.b[i, l] + raw.c[k, l])
			seller_total += share_s * rev
			carrier_total += (1.0 - share_s) * rev
			agents.append({"item": item, "agent": "bundle", "id": f"S{i + 1}-C{k + 1}", "bid": bid_b[i, k], "quantity": Qb, "payment": -rev, "unit_price": rev / Qb})
	qty = float(sum(q.values()))
	row.update({
		"p_plus": p_plus, "p_minus": p_minus, "qty_mech": qty,
		"welfare_mech": float(sum((bid_j[j] - bid_b[i, k]) * val for (j, i, k), val in q.items())),
		"n_buyers_trading": len(Q_j), "n_bundles_trading": len(Q_b),
		"buyer_payment_total": pay_total, "bundle_revenue_total": rev_total,
		"seller_revenue_total": seller_total, "carrier_revenue_total": carrier_total,
		"platform_margin": pay_total - rev_total,
		"buyer_unit_price_avg": pay_total / qty if qty > 0 else float("nan"),
		"bundle_unit_price_avg": rev_total / qty if qty > 0 else float("nan"),
		"ir_violations": ir_bad, "n_small_lps": n_lp,
	})
	return row, agents


def price_solution(cfg: Mapping[str, Any], raw: RawData, delta: int, sol: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
	# Loop over (slot, scenario, item) markets of one solved two-stage LP
	gp, GRB = import_gurobi()
	mkt = cfg_market(cfg)
	agg = aggregate(raw, cfg, int(delta))
	scen = cfg["scenarios"]
	L, J, K, T = len(mkt["items"]), mkt["J"], mkt["K"], agg["nslots"]
	xR, y, mvar = sol["xR"], sol["y"], sol["m"]
	rows: List[Dict[str, Any]] = []
	agent_rows: List[Dict[str, Any]] = []
	for s, sc in enumerate(scen):
		gsh = float(sc["gamma"])
		for t in range(T):
			used_w = {k: 0.0 for k in range(K)}
			used_v = {k: 0.0 for k in range(K)}
			y_by_l: List[Dict[Tuple[int, int, int], float]] = [{} for _ in range(L)]
			for (i, j, k, l, tt, ss), q in y.items():
				if tt != t or ss != s:
					continue
				y_by_l[l][j, i, k] = q
				used_w[k] += mkt["weight"][mkt["items"][l]] * q
				used_v[k] += mkt["volume"][mkt["items"][l]] * q
			for l, item in enumerate(mkt["items"]):
				w, v = mkt["weight"][item], mkt["volume"][item]
				other_w = {k: used_w[k] - w * sum(q for (j, i, kk), q in y_by_l[l].items() if kk == k) for k in range(K)}
				other_v = {k: used_v[k] - v * sum(q for (j, i, kk), q in y_by_l[l].items() if kk == k) for k in range(K)}
				cap_w = {k: max(0.0, (1.0 - gsh) * float(agg["cap_w"][k, t]) - other_w[k]) for k in range(K)}
				cap_v = {k: max(0.0, (1.0 - gsh) * float(agg["cap_v"][k, t]) - other_v[k]) for k in range(K)}
				buyers = [(j, float(raw.a[j, l]), float(agg["d_real"][j, l, t, s]) + (float(mvar[j, l, t - 1, s]) if t > 0 else 0.0)) for j in range(J)]
				bundles = [((i, k), float(raw.b[i, l] + raw.c[k, l]), float(xR.get((i, k, l, t), 0.0))) for i in mkt["sellers_by_item"][item] for k in range(K)]
				row, agents = price_market(gp, GRB, raw, mkt, l, buyers, bundles, cap_w, cap_v, y_by_l[l])
				head = {"scenario": sc["name"], "slot": t + 1}
				rows.append({**head, **row})
				for a in agents:
					agent_rows.append({**head, **a})
	return rows, agent_rows


def run_pricing(cfg: Mapping[str, Any], results_root: Path, seed: int, gammas: Sequence[float], f_costs: Sequence[float],
	h_scale: float, solver_overrides: Optional[Mapping[str, Any]]) -> None:
	# Price the recommended Delta* of each (Gamma, F) from stage3 for one seed
	rec_path = results_root / "05_stage3_regret" / "recommended_delta_by_gamma.csv"
	if not rec_path.exists():
		raise RuntimeError("Stage3 recommended_delta_by_gamma.csv not found. Run stage3 first.")
	rec = {}
	for r in read_rows(rec_path):
		rec[round(float(r["gamma"]), 6), round(float(r.get("f_cost") or 0.0), 6)] = int(float(r["delta_recommended"]))
	outdir = results_root / "07_pricing"
	outdir.mkdir(parents=True, exist_ok=True)
	raw = load_or_create_raw(cfg, results_root, int(seed))
	summary_rows: List[Dict[str, Any]] = []
	for gamma in gammas:
		for f_cost in f_costs:
			delta = rec.get((round(float(gamma), 6), round(float(f_cost), 6)))
			if delta is None:
				print(f"[pricing] no recommended Delta for Gamma={gamma}, F={f_cost}; skipped")
				continue
			t0 = time.time()
			sol: Dict[str, Any] = {}
			summ = solve_lp(cfg, raw, results_root, "pricing", int(seed), float(gamma), int(delta), float(h_scale),
				f_cost=float(f_cost), solution_out=sol, solver_overrides=solver_overrides)[0]
			rows, agents = price_solution(cfg, raw, int(delta), sol)
			stem = f"seed{seed}_G{sanitize_float(float(gamma))}_F{sanitize_float(float(f_cost))}_D{delta}"
			write_rows(outdir / f"markets_{stem}.csv", rows)
			write_rows(outdir / f"agents_{stem}.csv", agents)
			for sc in cfg["scenarios"]:
				sub = [r for r in rows if r["scenario"] == sc["name"]]
				qty = sum(r["qty_mech"] for r in sub)
				pay = sum(r["buyer_payment_total"] for r in sub)
				rev = sum(r["bundle_revenue_total"] for r in sub)
				w_lp = sum(r["welfare_lp"] for r in sub)
				summary_rows.append({
					"seed": seed, "gamma": gamma, "f_cost": f_cost, "delta": delta, "scenario": sc["name"],
					"qty_lp": sum(r["qty_lp"] for r in sub), "qty_mech": qty,
					"efficiency_qty": qty / sum(r["qty_lp"] for r in sub) if sub else float("nan"),
					"welfare_lp": w_lp, "welfare_mech": sum(r["welfare_mech"] for r in sub),
					"efficiency_welfare": sum(r["welfare_mech"] for r in sub) / w_lp if w_lp else float("nan"),
					"buyer_payment_total": pay, "bundle_revenue_total": rev,
					"seller_revenue_total": sum(r["seller_revenue_total"] for r in sub),
					"carrier_revenue_total": sum(r["carrier_revenue_total"] for r in sub),
					"platform_margin": pay - rev, "platform_margin_rate": (pay - rev) / pay if pay else float("nan"),
					"buyer_unit_price_avg": pay / qty if qty else float("nan"),
					"bundle_unit_price_avg": rev / qty if qty else float("nan"),
					"ir_violations": sum(r["ir_violations"] for r in sub),
					"budget_balanced": int(pay - rev >= -1e-6),
					"n_small_lps": sum(r["n_small_lps"] for r in sub), "runtime_sec": time.time() - t0,
				})
			print(f"[pricing] Gamma={gamma} F={f_cost} Delta*={delta}: {len(rows)} markets priced in {time.time() - t0:.1f}s")
	write_rows(outdir / "pricing_summary.csv", summary_rows)
	make_pricing_plots(outdir / "pricing_summary.csv", results_root / "figures" / "pricing")


def make_pricing_plots(path: Path, figdir: Path) -> None:
	# Platform margin rate and unit prices by scenario for each priced (Gamma, F)
	import pandas as pd
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	df = pd.read_csv(path)
	if df.empty:
		return
	figdir.mkdir(parents=True, exist_ok=True)
	df["label"] = df.apply(lambda r: f"G{r['gamma']:g}/F{r['f_cost']:g}/D{int(r['delta'])}", axis=1)
	for col, name in [("platform_margin_rate", "Platform margin rate"), ("buyer_unit_price_avg", "Buyer clearing price (avg)"), ("bundle_unit_price_avg", "Seller+carrier payment per unit (avg)"), ("efficiency_welfare", "Welfare efficiency vs LP")]:
		piv = df.pivot_table(index="label", columns="scenario", values=col)
		piv.plot(kind="bar", figsize=(9, 4.5))
		plt.ylabel(name)
		plt.title(f"MMTR-MVCG: {name}")
		plt.tight_layout()
		plt.savefig(figdir / f"{col}.png", dpi=180)
		plt.close()
