#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5PL Middle-mile Platform Research Pipeline (Model 9.2)
======================================================

Research order implemented from docs/tex/중견연구 통합모형_9.2.tex:
  check   : environment + deterministic raw-data check (no Gurobi solve)
  stage0  : Seed 101 sanity check
  stage1  : Baseline main experiment, 5 seeds x 5 Gamma x 28 Delta = 700 LPs
  stage2a : h^R sensitivity pilot, Seed 101 x Gamma {0,4,8} x 5 h-scales x Delta {1,7,14,28}
  stage2b : h^R sensitivity full Delta, Seed 101 x Gamma {0,4,8} x 5 h-scales x Delta 1..28
  stage3  : common auction cycle and Regret from stage1 results (no Gurobi solve)
  one     : solve one custom combination
  all     : stage0 -> stage1 -> stage2a -> stage3

Important implementation choices:
- Core model is a continuous two-stage robust LP. No z, minimum lot, or Big-M.
- Delta is an outer-loop design variable; a separate LP is solved for each Delta.
- The same raw data are reused for the same seed across Gamma, Delta, and h^R sensitivity.
- MMTR-MVCG pricing is NOT invented here; the model file only states it as post-processing.
  This pipeline exports positive-y trades so a pricing module can be attached later.

v14 extensions (config section "extensions"; all keys absent -> exact v13 Baseline LP):
- auction_fixed_cost F: -F * |T(Delta)| in the objective (per-auction operating cost).
- terminal_backlog_wait_slots kappa: horizon-end backlog pays c_wait * kappa * ell_T.
- service_level_min / service_shortfall_penalty: per-scenario soft service floor
  u_s >= SL * D_s - shipped_s, penalised by rho * u_s.
- F is an outer grid dimension like h_scale; stage4 (heuristics.py) and pricing
  (pricing.py, MMTR-MVCG after Guo et al. 2022) consume solve_lp via fixed_x / solution_out.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# Configuration and helpers
# -----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "config_baseline.json"
DEFAULT_RESULTS = SCRIPT_DIR / "results"


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_hash(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def ensure_dirs(root: Path) -> None:
    for sub in [
        "00_raw_data",
        "01_stage0_sanity",
        "02_stage1_baseline",
        "03_stage2_hcost_pilot",
        "04_stage2_hcost_full",
        "05_stage3_regret",
        "06_stage4_heuristics",
        "07_pricing",
        "logs",
        "iis",
        "figures",
    ]:
        (root / sub).mkdir(parents=True, exist_ok=True)


def parse_num_list(text: Optional[str], cast=float) -> Optional[List[Any]]:
    if text is None:
        return None
    out: List[Any] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part and part.count("-") == 1 and not part.startswith("-"):
            a, b = part.split("-", 1)
            if cast is int:
                aa, bb = int(a), int(b)
                step = 1 if bb >= aa else -1
                out.extend(range(aa, bb + step, step))
            else:
                raise ValueError("Ranges like 1-28 are supported only for integer lists.")
        else:
            out.append(cast(part))
    return out


def sanitize_float(x: float) -> str:
    s = f"{x:g}"
    return s.replace("-", "m").replace(".", "p")


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]], append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        if not path.exists() and not append:
            path.write_text("", encoding="utf-8-sig")
        return

    keys: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)

    # If appending to an existing non-empty CSV, preserve existing header/order.
    if append and path.exists() and path.stat().st_size > 0:
        with path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            existing = next(reader, [])
        extra = [k for k in keys if k not in existing]
        if extra:
            # Rewrite union schema to avoid malformed CSVs during incremental experiments.
            old = read_rows(path)
            union = existing + extra
            with path.open("w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=union)
                w.writeheader()
                for r in old:
                    w.writerow(r)
                for r in rows:
                    w.writerow(r)
            return
        with path.open("a", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=existing)
            for r in rows:
                w.writerow(r)
        return

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def is_finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


# -----------------------------------------------------------------------------
# Raw data
# -----------------------------------------------------------------------------

@dataclass
class RawData:
    demand: np.ndarray       # [J, L, R]
    supply: np.ndarray       # [I, L, R]
    cap_w: np.ndarray        # [K, R]
    cap_v: np.ndarray        # [K, R]
    a: np.ndarray            # [J, L]
    b: np.ndarray            # [I, L]
    c: np.ndarray            # [K, L]
    h: np.ndarray            # [I, K, L]
    xi: np.ndarray           # [J, L, R]
    metadata: Dict[str, Any]
    eps: Optional[np.ndarray] = None  # [J, L, R] v15 daily realization multiplier; None = deterministic (v13/v14)


def realized_daily_demand(raw: "RawData") -> np.ndarray:
    # Daily realized demand before scenario scaling: nominal x noise (v15) or nominal (v13/v14)
    return raw.demand if raw.eps is None else raw.demand * raw.eps


def cfg_market(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    m = cfg["market"]
    return {
        "I": int(m["n_sellers"]),
        "J": int(m["n_buyers"]),
        "K": int(m["n_carriers"]),
        "items": list(m["items"]),
        "sellers_by_item": {k: tuple(v) for k, v in m["sellers_by_item"].items()},
        "weight": {k: float(v) for k, v in m["weight"].items()},
        "volume": {k: float(v) for k, v in m["volume"].items()},
        "R": int(cfg["study"]["calendar_days"]),
    }


def item_map(cfg: Mapping[str, Any]) -> Dict[str, int]:
    return {item: i for i, item in enumerate(cfg["market"]["items"])}


def spawn_rngs(seed: int) -> Dict[str, np.random.Generator]:
    # Independent substreams as specified in the research design.
    # "realization" is appended last so the first four substreams are identical to v13/v14.
    names = ["demand", "supply_alloc", "capacity_alloc", "economics", "realization"]
    ss = np.random.SeedSequence(int(seed))
    children = ss.spawn(len(names))
    return {n: np.random.default_rng(ch) for n, ch in zip(names, children)}


def data_generation_fingerprint(cfg: Mapping[str, Any]) -> str:
    subset = {
        "study_days": cfg["study"]["calendar_days"],
        "market": cfg["market"],
        "data": cfg["data"],
        "economics": cfg["economics"],
    }
    return canonical_hash(subset)


def generate_raw_data(cfg: Mapping[str, Any], seed: int) -> RawData:
    m = cfg_market(cfg)
    dcfg = cfg["data"]
    ecfg = cfg["economics"]
    items = m["items"]
    imap = item_map(cfg)
    rng = spawn_rngs(seed)

    demand = rng["demand"].poisson(
        float(dcfg["demand_poisson_lambda"]),
        size=(m["J"], len(items), m["R"]),
    ).astype(float)

    supply = np.zeros((m["I"], len(items), m["R"]), dtype=float)
    for r in range(m["R"]):
        for item in items:
            l = imap[item]
            total_d = float(demand[:, l, r].sum())
            total_supply = float(math.ceil(float(dcfg["rho_supply"]) * total_d))
            sellers = m["sellers_by_item"][item]
            weights = rng["supply_alloc"].uniform(
                float(dcfg["seller_allocation_low"]),
                float(dcfg["seller_allocation_high"]),
                size=len(sellers),
            )
            shares = weights / weights.sum()
            for p, i in enumerate(sellers):
                supply[i, l, r] = total_supply * shares[p]

    cap_w = np.zeros((m["K"], m["R"]), dtype=float)
    cap_v = np.zeros((m["K"], m["R"]), dtype=float)
    for r in range(m["R"]):
        w_need = 0.0
        v_need = 0.0
        for item in items:
            l = imap[item]
            d_total = float(demand[:, l, r].sum())
            w_need += m["weight"][item] * d_total
            v_need += m["volume"][item] * d_total

        total_w = float(math.ceil(float(dcfg["rho_capacity"]) * w_need))
        total_v = float(math.ceil(float(dcfg["rho_capacity"]) * v_need))
        weights = rng["capacity_alloc"].uniform(
            float(dcfg["carrier_allocation_low"]),
            float(dcfg["carrier_allocation_high"]),
            size=m["K"],
        )
        shares = weights / weights.sum()
        cap_w[:, r] = total_w * shares
        cap_v[:, r] = total_v * shares

    erng = rng["economics"]
    a = erng.uniform(float(ecfg["a_low"]), float(ecfg["a_high"]), size=(m["J"], len(items)))
    b = erng.uniform(float(ecfg["b_low"]), float(ecfg["b_high"]), size=(m["I"], len(items)))
    c = erng.uniform(float(ecfg["c_low"]), float(ecfg["c_high"]), size=(m["K"], len(items)))
    h = erng.uniform(float(ecfg["h_low"]), float(ecfg["h_high"]), size=(m["I"], m["K"], len(items)))
    xi = float(dcfg["demand_deviation_rate"]) * demand

    # v15: independent daily multiplicative noise in [1-rate, 1+rate], shared by all execution scenarios,
    # so slot aggregation pools day-level deviations while realizations stay inside the budgeted set.
    # mode "market": one factor per (item, day) shared by all buyers (market-wide daily shock);
    # mode "buyer" : independent factor per (buyer, item, day), which already averages out across buyers at Delta=1.
    noise_rate = float(dcfg.get("realization_noise_rate", 0.0))
    noise_mode = str(dcfg.get("realization_noise_mode", "market"))
    eps = None
    if noise_rate > 0.0:
        if noise_mode == "buyer":
            eps = rng["realization"].uniform(1.0 - noise_rate, 1.0 + noise_rate, size=demand.shape)
        else:
            day_factor = rng["realization"].uniform(1.0 - noise_rate, 1.0 + noise_rate, size=(1, len(items), m["R"]))
            eps = np.broadcast_to(day_factor, demand.shape).copy()

    meta = {
        "seed": int(seed),
        "generation_fingerprint": data_generation_fingerprint(cfg),
        "config_version": cfg["study"].get("version", "9.2"),
    }
    return RawData(demand, supply, cap_w, cap_v, a, b, c, h, xi, meta, eps)


def raw_hash(raw: RawData) -> str:
    h = hashlib.sha256()
    arrays = [raw.demand, raw.supply, raw.cap_w, raw.cap_v, raw.a, raw.b, raw.c, raw.h, raw.xi]
    if raw.eps is not None:
        arrays.append(raw.eps)
    for arr in arrays:
        h.update(np.ascontiguousarray(arr).tobytes())
    h.update(json.dumps(raw.metadata, sort_keys=True).encode("utf-8"))
    return h.hexdigest()


def raw_paths(results_root: Path, seed: int) -> Tuple[Path, Path]:
    folder = results_root / "00_raw_data"
    return folder / f"seed_{seed}.npz", folder / f"seed_{seed}_metadata.json"


def save_raw(raw: RawData, results_root: Path, cfg: Mapping[str, Any]) -> None:
    npz, meta = raw_paths(results_root, int(raw.metadata["seed"]))
    extra = {"eps": raw.eps} if raw.eps is not None else {}
    np.savez_compressed(
        npz,
        demand=raw.demand,
        supply=raw.supply,
        cap_w=raw.cap_w,
        cap_v=raw.cap_v,
        a=raw.a,
        b=raw.b,
        c=raw.c,
        h=raw.h,
        xi=raw.xi,
        **extra,
    )
    payload = dict(raw.metadata)
    payload["raw_sha256"] = raw_hash(raw)
    payload["data_generation_parameters"] = {
        "market": cfg["market"],
        "data": cfg["data"],
        "economics": cfg["economics"],
    }
    meta.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_or_create_raw(
    cfg: Mapping[str, Any],
    results_root: Path,
    seed: int,
    overwrite: bool = False,
) -> RawData:
    npz_path, meta_path = raw_paths(results_root, seed)
    expected_fp = data_generation_fingerprint(cfg)

    if npz_path.exists() and meta_path.exists() and not overwrite:
        meta = load_json(meta_path)
        if meta.get("generation_fingerprint") != expected_fp:
            raise RuntimeError(
                f"Cached raw data for seed={seed} were generated with a different config.\n"
                f"Delete {npz_path} and {meta_path}, or rerun with --overwrite-data."
            )
        z = np.load(npz_path)
        raw = RawData(
            demand=z["demand"], supply=z["supply"], cap_w=z["cap_w"], cap_v=z["cap_v"],
            a=z["a"], b=z["b"], c=z["c"], h=z["h"], xi=z["xi"],
            metadata={
                "seed": int(seed),
                "generation_fingerprint": expected_fp,
                "config_version": cfg["study"].get("version", "9.2"),
            },
            eps=z["eps"] if "eps" in z.files else None,
        )
        if meta.get("raw_sha256") and meta["raw_sha256"] != raw_hash(raw):
            raise RuntimeError(f"Raw data hash mismatch for seed={seed}; cached file may be corrupted.")
        return raw

    raw = generate_raw_data(cfg, seed)
    save_raw(raw, results_root, cfg)
    return raw


def save_daily_summary(raw: RawData, cfg: Mapping[str, Any], results_root: Path) -> None:
    m = cfg_market(cfg)
    imap = item_map(cfg)
    rows: List[Dict[str, Any]] = []
    for r in range(m["R"]):
        row: Dict[str, Any] = {"seed": raw.metadata["seed"], "day": r + 1}
        w_need = 0.0
        v_need = 0.0
        for item in m["items"]:
            l = imap[item]
            d = float(raw.demand[:, l, r].sum())
            s = float(raw.supply[:, l, r].sum())
            row[f"demand_{item}"] = d
            row[f"realized_demand_{item}"] = float(realized_daily_demand(raw)[:, l, r].sum())
            row[f"supply_{item}"] = s
            row[f"supply_demand_ratio_{item}"] = s / d if d > 0 else math.nan
            w_need += m["weight"][item] * d
            v_need += m["volume"][item] * d
        total_w = float(raw.cap_w[:, r].sum())
        total_v = float(raw.cap_v[:, r].sum())
        row.update({
            "weight_need": w_need,
            "weight_capacity": total_w,
            "weight_nominal_util": w_need / total_w if total_w > 0 else math.nan,
            "volume_need": v_need,
            "volume_capacity": total_v,
            "volume_nominal_util": v_need / total_v if total_v > 0 else math.nan,
        })
        rows.append(row)
    out = results_root / "00_raw_data" / f"seed_{raw.metadata['seed']}_daily_summary.csv"
    write_rows(out, rows)


def aggregate(raw: RawData, cfg: Mapping[str, Any], delta: int) -> Dict[str, Any]:
    m = cfg_market(cfg)
    nslots = int(math.ceil(m["R"] / int(delta)))
    day_groups: List[np.ndarray] = []
    for t in range(nslots):
        start = t * int(delta)
        end = min((t + 1) * int(delta), m["R"])
        day_groups.append(np.arange(start, end, dtype=int))

    L = len(m["items"])
    d_hat = np.zeros((m["J"], L, nslots), dtype=float)
    xi = np.zeros_like(d_hat)
    s_hat = np.zeros((m["I"], L, nslots), dtype=float)
    cap_w = np.zeros((m["K"], nslots), dtype=float)
    cap_v = np.zeros((m["K"], nslots), dtype=float)
    ell = np.zeros(nslots, dtype=float)
    tau = np.zeros(nslots, dtype=float)

    for t, days in enumerate(day_groups):
        d_hat[:, :, t] = raw.demand[:, :, days].sum(axis=2)
        xi[:, :, t] = raw.xi[:, :, days].sum(axis=2)
        s_hat[:, :, t] = raw.supply[:, :, days].sum(axis=2)
        cap_w[:, t] = raw.cap_w[:, days].sum(axis=1)
        cap_v[:, t] = raw.cap_v[:, days].sum(axis=1)
        ell[t] = float(len(days))
        tau[t] = (ell[t] - 1.0) / 2.0

    # Realized slot demand pools daily realizations (identical to d_hat when no v15 noise).
    daily_real = realized_daily_demand(raw)
    d_base = np.zeros_like(d_hat)
    for t, days in enumerate(day_groups):
        d_base[:, :, t] = daily_real[:, :, days].sum(axis=2)

    scenarios = cfg["scenarios"]
    d_real = np.zeros((m["J"], L, nslots, len(scenarios)), dtype=float)
    for s, sc in enumerate(scenarios):
        d_real[:, :, :, s] = (1.0 + float(sc["theta"])) * d_base

    return {
        "nslots": nslots,
        "day_groups": day_groups,
        "d_hat": d_hat,
        "xi": xi,
        "s_hat": s_hat,
        "cap_w": cap_w,
        "cap_v": cap_v,
        "ell": ell,
        "tau": tau,
        "d_real": d_real,
    }


def theoretical_bs_buffer(xi_vec: np.ndarray, gamma_budget: float) -> float:
    vals = sorted((float(x) for x in xi_vec), reverse=True)
    g = max(0.0, min(float(gamma_budget), float(len(vals))))
    whole = int(math.floor(g))
    frac = g - whole
    ans = sum(vals[:whole])
    if frac > 1e-12 and whole < len(vals):
        ans += frac * vals[whole]
    return float(ans)


def raw_diagnostics(raw: RawData, cfg: Mapping[str, Any]) -> Dict[str, Any]:
    m = cfg_market(cfg)
    imap = item_map(cfg)
    out: Dict[str, Any] = {
        "seed": raw.metadata["seed"],
        "raw_sha256": raw_hash(raw),
    }
    for item in m["items"]:
        l = imap[item]
        daily_d = raw.demand[:, l, :].sum(axis=0)
        daily_s = raw.supply[:, l, :].sum(axis=0)
        out[f"mean_daily_demand_{item}"] = float(daily_d.mean())
        out[f"mean_daily_supply_{item}"] = float(daily_s.mean())
        out[f"mean_supply_demand_ratio_{item}"] = float(np.mean(daily_s / daily_d))
    w_need = np.zeros(m["R"], dtype=float)
    v_need = np.zeros(m["R"], dtype=float)
    for item in m["items"]:
        l = imap[item]
        td = raw.demand[:, l, :].sum(axis=0)
        w_need += m["weight"][item] * td
        v_need += m["volume"][item] * td
    out["mean_nominal_weight_util"] = float(np.mean(w_need / raw.cap_w.sum(axis=0)))
    out["mean_nominal_volume_util"] = float(np.mean(v_need / raw.cap_v.sum(axis=0)))
    return out


# -----------------------------------------------------------------------------
# Gurobi model
# -----------------------------------------------------------------------------


def ext_params(cfg: Mapping[str, Any]) -> Dict[str, float]:
    # v14 extension parameters; every key absent reproduces the v13 Baseline LP exactly.
    e = cfg.get("extensions", {})
    return {
        "f_default": float(e.get("auction_fixed_cost_default", 0.0)),
        "kappa": float(e.get("terminal_backlog_wait_slots", 0.0)),
        "sl_min": float(e.get("service_level_min", 0.0)),
        "rho": float(e.get("service_shortfall_penalty", 0.0)),
    }


def f_cost_grid(cfg: Mapping[str, Any]) -> List[float]:
    e = cfg.get("extensions", {})
    grid = e.get("auction_fixed_cost_grid")
    if grid:
        return [float(x) for x in grid]
    return [ext_params(cfg)["f_default"]]


def import_gurobi():
    try:
        import gurobipy as gp
        from gurobipy import GRB
        return gp, GRB
    except Exception as e:
        raise RuntimeError(
            "gurobipy를 불러오지 못했습니다. requirements.txt 설치 후 Gurobi license를 설정하세요.\n"
            f"원래 오류: {e}"
        ) from e


def status_name(GRB, status: int) -> str:
    names = {
        GRB.LOADED: "LOADED", GRB.OPTIMAL: "OPTIMAL", GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD", GRB.UNBOUNDED: "UNBOUNDED",
        GRB.CUTOFF: "CUTOFF", GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
        GRB.NODE_LIMIT: "NODE_LIMIT", GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT", GRB.INTERRUPTED: "INTERRUPTED",
        GRB.NUMERIC: "NUMERIC", GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
    }
    return names.get(status, f"STATUS_{status}")


def specialized_x_keys(cfg: Mapping[str, Any], nslots: int) -> Iterable[Tuple[int, int, int, int]]:
    m = cfg_market(cfg)
    imap = item_map(cfg)
    for t in range(nslots):
        for item in m["items"]:
            l = imap[item]
            for i in m["sellers_by_item"][item]:
                for k in range(m["K"]):
                    yield i, k, l, t


def trade_arcs(cfg: Mapping[str, Any]) -> List[Tuple[int, int, int, int]]:
    m = cfg_market(cfg)
    imap = item_map(cfg)
    arcs: List[Tuple[int, int, int, int]] = []
    for item in m["items"]:
        l = imap[item]
        for i in m["sellers_by_item"][item]:
            for j in range(m["J"]):
                for k in range(m["K"]):
                    arcs.append((i, j, k, l))
    return arcs


def solve_lp(
    cfg: Mapping[str, Any],
    raw: RawData,
    results_root: Path,
    stage_name: str,
    seed: int,
    gamma_budget: float,
    delta: int,
    h_scale: float = 1.0,
    f_cost: float = 0.0,
    fixed_x: Optional[Mapping[Tuple[int, int, int, int], float]] = None,
    solution_out: Optional[Dict[str, Any]] = None,
    export_positive_trades: bool = False,
    solver_overrides: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    # fixed_x: evaluate a given first-stage contract with exact LP recourse (robust cover dropped).
    # solution_out: if a dict is passed, xR / y / m values are stored into it for post-processing.
    gp, GRB = import_gurobi()
    mkt = cfg_market(cfg)
    imap = item_map(cfg)
    ext = ext_params(cfg)
    kappa, sl_min, rho = ext["kappa"], ext["sl_min"], ext["rho"]
    agg = aggregate(raw, cfg, int(delta))
    nslots = agg["nslots"]
    d_hat = agg["d_hat"]
    xi = agg["xi"]
    s_hat = agg["s_hat"]
    cap_w = agg["cap_w"]
    cap_v = agg["cap_v"]
    ell = agg["ell"]
    tau = agg["tau"]
    d_real = agg["d_real"]
    scenarios = cfg["scenarios"]
    c_wait = float(cfg["economics"]["c_wait"])
    arcs = trade_arcs(cfg)

    solver_cfg = dict(cfg["solver"])
    if solver_overrides:
        solver_cfg.update(solver_overrides)

    model_name = f"{stage_name}_seed{seed}_G{sanitize_float(gamma_budget)}_D{delta}_h{sanitize_float(h_scale)}_F{sanitize_float(f_cost)}"
    model = gp.Model(model_name)
    model.Params.OutputFlag = int(solver_cfg.get("output_flag", 0))
    model.Params.Threads = int(solver_cfg.get("threads", 4))
    model.Params.TimeLimit = float(solver_cfg.get("time_limit_sec", 600.0))
    model.Params.NumericFocus = int(solver_cfg.get("numeric_focus", 1))
    method = int(solver_cfg.get("method", -1))
    if method >= 0:
        model.Params.Method = method
    if bool(solver_cfg.get("write_log_files", False)):
        logdir = results_root / "logs" / stage_name
        logdir.mkdir(parents=True, exist_ok=True)
        model.Params.LogFile = str(logdir / f"{model_name}.log")

    # Variables ----------------------------------------------------------------
    xR: Dict[Tuple[int, int, int, int], Any] = {}
    for i, k, l, t in specialized_x_keys(cfg, nslots):
        xR[i, k, l, t] = model.addVar(lb=0.0, name=f"xR[{i},{k},{l},{t}]")

    eta = {(l, t): model.addVar(lb=0.0, name=f"eta[{l},{t}]")
           for l in range(len(mkt["items"])) for t in range(nslots)}
    mu = {(j, l, t): model.addVar(lb=0.0, name=f"mu[{j},{l},{t}]")
          for j in range(mkt["J"]) for l in range(len(mkt["items"])) for t in range(nslots)}
    mvar = {(j, l, t, s): model.addVar(lb=0.0, name=f"m[{j},{l},{t},{s}]")
            for j in range(mkt["J"]) for l in range(len(mkt["items"]))
            for t in range(nslots) for s in range(len(scenarios))}
    y = {(i, j, k, l, t, s): model.addVar(lb=0.0, name=f"y[{i},{j},{k},{l},{t},{s}]")
         for (i, j, k, l) in arcs for t in range(nslots) for s in range(len(scenarios))}
    u = {s: model.addVar(lb=0.0, name=f"u[{s}]") for s in range(len(scenarios))}
    if fixed_x is not None:
        for key, var in xR.items():
            val = max(0.0, float(fixed_x.get(key, 0.0)))
            var.LB = val
            var.UB = val
    model.update()

    # Objective -----------------------------------------------------------------
    precontract_cost = gp.quicksum(
        float(h_scale) * raw.h[i, k, l] * var for (i, k, l, t), var in xR.items()
    )
    auction_cost = float(f_cost) * float(nslots)

    scenario_exprs = []
    slot_wait_constants: Dict[int, float] = {}
    for s, sc in enumerate(scenarios):
        trade_expr = gp.quicksum(
            (raw.a[j, l] - raw.b[i, l] - raw.c[k, l]) * y[i, j, k, l, t, s]
            for (i, j, k, l) in arcs for t in range(nslots)
        )
        slot_wait = float(
            sum(
                c_wait * tau[t] * float(d_real[j, l, t, s])
                for t in range(nslots)
                for j in range(mkt["J"])
                for l in range(len(mkt["items"]))
            )
        )
        slot_wait_constants[s] = slot_wait

        # Exactly follows sum_t c_wait * ell_t * m_{t-1}; initial m_0 is zero.
        backlog_wait = gp.quicksum(
            c_wait * ell[t] * mvar[j, l, t - 1, s]
            for t in range(1, nslots)
            for j in range(mkt["J"])
            for l in range(len(mkt["items"]))
        )
        # v14: horizon-end backlog waits kappa more slots of the final slot length.
        terminal_wait = gp.quicksum(
            c_wait * kappa * ell[nslots - 1] * mvar[j, l, nslots - 1, s]
            for j in range(mkt["J"])
            for l in range(len(mkt["items"]))
        )
        scenario_exprs.append(
            float(sc["weight"]) * (trade_expr - slot_wait - backlog_wait - terminal_wait - rho * u[s])
        )

    model.setObjective(-precontract_cost - auction_cost + gp.quicksum(scenario_exprs), GRB.MAXIMIZE)

    # v14: per-scenario soft service floor u_s >= SL * D_s - shipped_s
    for s in range(len(scenarios)):
        total_d = float(d_real[:, :, :, s].sum())
        model.addConstr(
            u[s] >= sl_min * total_d - gp.quicksum(y[i, j, k, l, t, s] for (i, j, k, l) in arcs for t in range(nslots)),
            name=f"service_soft[{s}]",
        )

    # Constraints: first-stage robust cover ------------------------------------
    for l, item in enumerate(mkt["items"]):
        sellers = mkt["sellers_by_item"][item]
        for t in range(nslots):
            if fixed_x is not None:
                break
            model.addConstr(
                gp.quicksum(xR[i, k, l, t] for i in sellers for k in range(mkt["K"]))
                >= float(d_hat[:, l, t].sum())
                + float(gamma_budget) * eta[l, t]
                + gp.quicksum(mu[j, l, t] for j in range(mkt["J"])),
                name=f"robust_cover[{l},{t}]",
            )
            for j in range(mkt["J"]):
                model.addConstr(
                    eta[l, t] + mu[j, l, t] >= float(xi[j, l, t]),
                    name=f"robust_dual[{j},{l},{t}]",
                )

    # First-stage seller supply
    for l, item in enumerate(mkt["items"]):
        for i in mkt["sellers_by_item"][item]:
            for t in range(nslots):
                model.addConstr(
                    gp.quicksum(xR[i, k, l, t] for k in range(mkt["K"])) <= float(s_hat[i, l, t]),
                    name=f"pre_supply[{i},{l},{t}]",
                )

    # First-stage W/V capacity
    for k in range(mkt["K"]):
        for t in range(nslots):
            model.addConstr(
                gp.quicksum(
                    mkt["weight"][item] * xR[i, k, imap[item], t]
                    for item in mkt["items"] for i in mkt["sellers_by_item"][item]
                ) <= float(cap_w[k, t]),
                name=f"pre_weight[{k},{t}]",
            )
            model.addConstr(
                gp.quicksum(
                    mkt["volume"][item] * xR[i, k, imap[item], t]
                    for item in mkt["items"] for i in mkt["sellers_by_item"][item]
                ) <= float(cap_v[k, t]),
                name=f"pre_volume[{k},{t}]",
            )

    # Second-stage demand balance
    for s in range(len(scenarios)):
        for t in range(nslots):
            for l, item in enumerate(mkt["items"]):
                sellers = mkt["sellers_by_item"][item]
                for j in range(mkt["J"]):
                    shipped = gp.quicksum(y[i, j, k, l, t, s] for i in sellers for k in range(mkt["K"]))
                    prev = 0.0 if t == 0 else mvar[j, l, t - 1, s]
                    model.addConstr(
                        shipped + mvar[j, l, t, s] == float(d_real[j, l, t, s]) + prev,
                        name=f"demand_balance[{j},{l},{t},{s}]",
                    )

    # Contract-use limit
    for s in range(len(scenarios)):
        for t in range(nslots):
            for l, item in enumerate(mkt["items"]):
                for i in mkt["sellers_by_item"][item]:
                    for k in range(mkt["K"]):
                        model.addConstr(
                            gp.quicksum(y[i, j, k, l, t, s] for j in range(mkt["J"])) <= xR[i, k, l, t],
                            name=f"contract_use[{i},{k},{l},{t},{s}]",
                        )

    # Execution W/V capacity under network disruption
    for s, sc in enumerate(scenarios):
        gshock = float(sc["gamma"])
        for k in range(mkt["K"]):
            for t in range(nslots):
                model.addConstr(
                    gp.quicksum(
                        mkt["weight"][mkt["items"][l]] * y[i, j, k, l, t, s]
                        for (i, j, kk, l) in arcs if kk == k
                    ) <= (1.0 - gshock) * float(cap_w[k, t]),
                    name=f"run_weight[{k},{t},{s}]",
                )
                model.addConstr(
                    gp.quicksum(
                        mkt["volume"][mkt["items"][l]] * y[i, j, k, l, t, s]
                        for (i, j, kk, l) in arcs if kk == k
                    ) <= (1.0 - gshock) * float(cap_v[k, t]),
                    name=f"run_volume[{k},{t},{s}]",
                )

    # Optimize ------------------------------------------------------------------
    start_wall = time.time()
    model.optimize()
    wall = time.time() - start_wall
    if model.Status == GRB.INF_OR_UNBD:
        model.Params.DualReductions = 0
        model.optimize()

    st = status_name(GRB, model.Status)
    has_solution = int(model.SolCount > 0)

    if model.Status == GRB.INFEASIBLE and bool(solver_cfg.get("compute_iis_on_infeasible", True)):
        iis_dir = results_root / "iis" / stage_name
        iis_dir.mkdir(parents=True, exist_ok=True)
        model.computeIIS()
        model.write(str(iis_dir / f"{model_name}.ilp"))

    summary: Dict[str, Any] = {
        "stage": stage_name,
        "seed": int(seed),
        "raw_sha256": raw_hash(raw),
        "gamma": float(gamma_budget),
        "delta": int(delta),
        "h_scale": float(h_scale),
        "f_cost": float(f_cost),
        "status": st,
        "has_solution": has_solution,
        "runtime_sec": float(getattr(model, "Runtime", wall)),
        "wall_sec": float(wall),
        "simplex_iterations": float(getattr(model, "IterCount", 0.0)),
        "barrier_iterations": float(getattr(model, "BarIterCount", 0.0)),
        "num_vars": int(model.NumVars),
        "num_constrs": int(model.NumConstrs),
    }

    robust_rows: List[Dict[str, Any]] = []
    scenario_rows: List[Dict[str, Any]] = []
    capacity_rows: List[Dict[str, Any]] = []
    trade_rows: List[Dict[str, Any]] = []

    if not has_solution:
        summary.update({"objective": math.nan, "precontract_qty": math.nan})
        return summary, robust_rows, scenario_rows, capacity_rows, trade_rows

    summary["objective"] = float(model.ObjVal)

    # Objective decomposition ----------------------------------------------------
    pre_qty = float(sum(v.X for v in xR.values()))
    pre_cost = float(sum(float(h_scale) * raw.h[i, k, l] * v.X for (i, k, l, t), v in xR.items()))
    summary["precontract_qty"] = pre_qty
    summary["precontract_cost"] = pre_cost
    summary["auction_cost"] = auction_cost
    summary["n_auctions"] = int(nslots)

    weighted_trade = 0.0
    weighted_slot_wait = 0.0
    weighted_backlog_wait = 0.0
    weighted_terminal = 0.0
    weighted_service_pen = 0.0
    weighted_demand = 0.0
    weighted_shipped = 0.0
    weighted_final_backlog = 0.0

    for s, sc in enumerate(scenarios):
        shipped = 0.0
        trade_surplus = 0.0
        for (i, j, k, l) in arcs:
            for t in range(nslots):
                val = float(y[i, j, k, l, t, s].X)
                shipped += val
                trade_surplus += float(raw.a[j, l] - raw.b[i, l] - raw.c[k, l]) * val
                if export_positive_trades and val > 1e-8:
                    trade_rows.append({
                        "stage": stage_name, "seed": seed, "gamma": gamma_budget,
                        "delta": delta, "h_scale": h_scale, "scenario": sc["name"],
                        "slot": t + 1, "seller": i + 1, "buyer": j + 1,
                        "carrier": k + 1, "item": mkt["items"][l], "quantity": val,
                        "a": float(raw.a[j, l]), "b": float(raw.b[i, l]), "c": float(raw.c[k, l]),
                        "unit_base_surplus": float(raw.a[j, l] - raw.b[i, l] - raw.c[k, l]),
                    })

        realized = float(d_real[:, :, :, s].sum())
        backlog_sum = float(sum(mvar[j, l, t, s].X
                                for j in range(mkt["J"]) for l in range(len(mkt["items"]))
                                for t in range(nslots)))
        final_backlog = float(sum(mvar[j, l, nslots - 1, s].X
                                  for j in range(mkt["J"]) for l in range(len(mkt["items"]))))
        backlog_wait = float(sum(
            c_wait * ell[t] * mvar[j, l, t - 1, s].X
            for t in range(1, nslots)
            for j in range(mkt["J"])
            for l in range(len(mkt["items"]))
        ))
        slot_wait = float(slot_wait_constants[s])
        service = shipped / realized if realized > 0 else math.nan
        terminal_cost = c_wait * kappa * float(ell[nslots - 1]) * final_backlog
        shortfall = float(u[s].X)
        service_pen = rho * shortfall

        scenario_rows.append({
            "stage": stage_name, "seed": seed, "gamma": gamma_budget, "delta": delta,
            "h_scale": h_scale, "scenario": sc["name"], "weight": float(sc["weight"]),
            "theta": float(sc["theta"]), "network_gamma": float(sc["gamma"]),
            "realized_demand": realized, "shipped": shipped, "service_rate": service,
            "backlog_sum": backlog_sum, "final_backlog": final_backlog,
            "trade_surplus": trade_surplus, "slot_wait_cost": slot_wait,
            "backlog_wait_cost": backlog_wait,
            "terminal_backlog_cost": terminal_cost,
            "service_shortfall": shortfall,
            "service_penalty": service_pen,
            "scenario_operating_term": trade_surplus - slot_wait - backlog_wait - terminal_cost - service_pen,
        })

        summary[f"service_{sc['name']}"] = service
        summary[f"final_backlog_{sc['name']}"] = final_backlog
        summary[f"service_shortfall_{sc['name']}"] = shortfall
        wgt = float(sc["weight"])
        weighted_trade += wgt * trade_surplus
        weighted_slot_wait += wgt * slot_wait
        weighted_backlog_wait += wgt * backlog_wait
        weighted_terminal += wgt * terminal_cost
        weighted_service_pen += wgt * service_pen
        weighted_demand += wgt * realized
        weighted_shipped += wgt * shipped
        weighted_final_backlog += wgt * final_backlog

    summary["weighted_trade_surplus"] = weighted_trade
    summary["weighted_slot_wait_cost"] = weighted_slot_wait
    summary["weighted_backlog_wait_cost"] = weighted_backlog_wait
    summary["weighted_terminal_backlog_cost"] = weighted_terminal
    summary["weighted_service_penalty"] = weighted_service_pen
    summary["weighted_service_rate"] = weighted_shipped / weighted_demand if weighted_demand > 0 else math.nan
    summary["weighted_final_backlog"] = weighted_final_backlog
    summary["objective_reconstructed"] = (
        -pre_cost - auction_cost + weighted_trade - weighted_slot_wait - weighted_backlog_wait
        - weighted_terminal - weighted_service_pen
    )
    summary["objective_reconstruction_error"] = summary["objective"] - summary["objective_reconstructed"]

    # Robust diagnostics ---------------------------------------------------------
    for l, item in enumerate(mkt["items"]):
        sellers = mkt["sellers_by_item"][item]
        for t in range(nslots):
            nominal = float(d_hat[:, l, t].sum())
            theory_buffer = theoretical_bs_buffer(xi[:, l, t], gamma_budget)
            theory_target = nominal + theory_buffer
            eta_val = float(eta[l, t].X)
            mu_sum = float(sum(mu[j, l, t].X for j in range(mkt["J"])))
            model_rhs = nominal + float(gamma_budget) * eta_val + mu_sum
            x_total = float(sum(xR[i, k, l, t].X for i in sellers for k in range(mkt["K"])))
            robust_rows.append({
                "stage": stage_name, "seed": seed, "gamma": gamma_budget, "delta": delta,
                "h_scale": h_scale, "item": item, "slot": t + 1,
                "slot_length_days": float(ell[t]), "nominal_demand": nominal,
                "theoretical_buffer": theory_buffer, "theoretical_target": theory_target,
                "eta": eta_val, "mu_sum": mu_sum, "model_robust_rhs": model_rhs,
                "precontract_qty": x_total, "slack_vs_theory": x_total - theory_target,
                "slack_vs_model_rhs": x_total - model_rhs,
                "seller_supply": float(s_hat[:, l, t].sum()),
            })

    # Capacity diagnostics -------------------------------------------------------
    tol = 1e-7
    pre_w_bind = pre_v_bind = 0
    pre_total = mkt["K"] * nslots
    run_w_bind = run_v_bind = 0
    run_total = mkt["K"] * nslots * len(scenarios)

    for k in range(mkt["K"]):
        for t in range(nslots):
            used_w = 0.0
            used_v = 0.0
            for item in mkt["items"]:
                l = imap[item]
                for i in mkt["sellers_by_item"][item]:
                    q = float(xR[i, k, l, t].X)
                    used_w += mkt["weight"][item] * q
                    used_v += mkt["volume"][item] * q
            uw = used_w / float(cap_w[k, t]) if cap_w[k, t] > 0 else math.nan
            uv = used_v / float(cap_v[k, t]) if cap_v[k, t] > 0 else math.nan
            pre_w_bind += int(is_finite(uw) and uw >= 1.0 - 1e-6)
            pre_v_bind += int(is_finite(uv) and uv >= 1.0 - 1e-6)
            capacity_rows.append({
                "stage": stage_name, "seed": seed, "gamma": gamma_budget, "delta": delta,
                "h_scale": h_scale, "phase": "precontract", "scenario": "",
                "carrier": k + 1, "slot": t + 1,
                "weight_used": used_w, "weight_capacity": float(cap_w[k, t]), "weight_util": uw,
                "volume_used": used_v, "volume_capacity": float(cap_v[k, t]), "volume_util": uv,
            })

            for s, sc in enumerate(scenarios):
                used_w_s = 0.0
                used_v_s = 0.0
                for (i, j, kk, l) in arcs:
                    if kk != k:
                        continue
                    item = mkt["items"][l]
                    q = float(y[i, j, k, l, t, s].X)
                    used_w_s += mkt["weight"][item] * q
                    used_v_s += mkt["volume"][item] * q
                avail_w = (1.0 - float(sc["gamma"])) * float(cap_w[k, t])
                avail_v = (1.0 - float(sc["gamma"])) * float(cap_v[k, t])
                uw_s = used_w_s / avail_w if avail_w > 0 else math.nan
                uv_s = used_v_s / avail_v if avail_v > 0 else math.nan
                run_w_bind += int(is_finite(uw_s) and uw_s >= 1.0 - 1e-6)
                run_v_bind += int(is_finite(uv_s) and uv_s >= 1.0 - 1e-6)
                capacity_rows.append({
                    "stage": stage_name, "seed": seed, "gamma": gamma_budget, "delta": delta,
                    "h_scale": h_scale, "phase": "execution", "scenario": sc["name"],
                    "carrier": k + 1, "slot": t + 1,
                    "weight_used": used_w_s, "weight_capacity": avail_w, "weight_util": uw_s,
                    "volume_used": used_v_s, "volume_capacity": avail_v, "volume_util": uv_s,
                })

    summary["pre_weight_binding_rate"] = pre_w_bind / pre_total if pre_total else math.nan
    summary["pre_volume_binding_rate"] = pre_v_bind / pre_total if pre_total else math.nan
    summary["run_weight_binding_rate"] = run_w_bind / run_total if run_total else math.nan
    summary["run_volume_binding_rate"] = run_v_bind / run_total if run_total else math.nan

    if solution_out is not None:
        solution_out["xR"] = {key: float(v.X) for key, v in xR.items()}
        solution_out["y"] = {key: float(v.X) for key, v in y.items() if v.X > 1e-9}
        solution_out["m"] = {key: float(v.X) for key, v in mvar.items()}
        solution_out["u"] = {s: float(v.X) for s, v in u.items()}

    return summary, robust_rows, scenario_rows, capacity_rows, trade_rows


# -----------------------------------------------------------------------------
# Experiment runner / resume support
# -----------------------------------------------------------------------------


def stage_folder(results_root: Path, stage: str) -> Path:
    return {
        "stage0": results_root / "01_stage0_sanity",
        "stage1": results_root / "02_stage1_baseline",
        "stage2a": results_root / "03_stage2_hcost_pilot",
        "stage2b": results_root / "04_stage2_hcost_full",
    }[stage]


def run_key(seed: int, gamma: float, delta: int, h_scale: float, f_cost: float = 0.0) -> Tuple[int, float, int, float, float]:
    return int(seed), round(float(gamma), 10), int(delta), round(float(h_scale), 10), round(float(f_cost), 10)


def completed_keys(summary_path: Path) -> set:
    keys = set()
    for r in read_rows(summary_path):
        try:
            if int(float(r.get("has_solution", "0"))) == 1:
                keys.add(run_key(int(r["seed"]), float(r["gamma"]), int(r["delta"]), float(r["h_scale"]),
                                 float(r.get("f_cost") or 0.0)))
        except Exception:
            continue
    return keys


def run_grid(
    cfg: Mapping[str, Any],
    results_root: Path,
    stage: str,
    seeds: Sequence[int],
    gammas: Sequence[float],
    deltas: Sequence[int],
    h_scales: Sequence[float],
    overwrite_data: bool,
    resume: bool,
    export_positive_trades: bool = False,
    solver_overrides: Optional[Mapping[str, Any]] = None,
    f_costs: Optional[Sequence[float]] = None,
) -> None:
    f_costs = [float(x) for x in f_costs] if f_costs else [0.0]
    folder = stage_folder(results_root, stage)
    folder.mkdir(parents=True, exist_ok=True)
    summary_path = folder / "run_summary.csv"
    robust_path = folder / "robust_diagnostics.csv"
    scenario_path = folder / "scenario_summary.csv"
    capacity_path = folder / "capacity_diagnostics.csv"
    trade_path = folder / "positive_trades.csv"

    if not resume:
        for pth in [summary_path, robust_path, scenario_path, capacity_path, trade_path]:
            if pth.exists():
                pth.unlink()
    done = completed_keys(summary_path) if resume else set()
    total = len(seeds) * len(gammas) * len(deltas) * len(h_scales) * len(f_costs)
    idx = 0
    new_count = 0

    for seed in seeds:
        raw = load_or_create_raw(cfg, results_root, int(seed), overwrite=overwrite_data)
        save_daily_summary(raw, cfg, results_root)
        for gamma in gammas:
            for h_scale in h_scales:
                for f_cost in f_costs:
                    for delta in deltas:
                        idx += 1
                        key = run_key(seed, gamma, delta, h_scale, f_cost)
                        tag = f"seed={seed}, Gamma={gamma}, Delta={delta}, h={h_scale}, F={f_cost}"
                        if key in done:
                            print(f"[{idx}/{total}] SKIP completed {tag}")
                            continue
                        print(f"[{idx}/{total}] SOLVE {tag}")
                        summary, robust, scen, cap, trades = solve_lp(
                            cfg, raw, results_root, stage, int(seed), float(gamma), int(delta), float(h_scale),
                            f_cost=float(f_cost),
                            export_positive_trades=export_positive_trades,
                            solver_overrides=solver_overrides,
                        )
                        write_rows(summary_path, [summary], append=True)
                        write_rows(robust_path, robust, append=True)
                        write_rows(scenario_path, scen, append=True)
                        write_rows(capacity_path, cap, append=True)
                        if export_positive_trades:
                            write_rows(trade_path, trades, append=True)
                        new_count += 1
                        print(
                            f"    status={summary['status']} obj={summary.get('objective')} "
                            f"xR={summary.get('precontract_qty')} SR={summary.get('weighted_service_rate')} "
                            f"runtime={summary.get('runtime_sec'):.3f}s"
                        )

    print(f"\n[{stage}] complete. Newly solved: {new_count}. Results: {folder}")
    make_stage_plots(summary_path, results_root / "figures" / stage, stage)
    if stage == "stage0":
        build_sanity_report(summary_path, robust_path, scenario_path, capacity_path, folder / "sanity_report.txt")


def make_stage_plots(summary_path: Path, figdir: Path, stage: str) -> None:
    rows = read_rows(summary_path)
    if not rows:
        return
    try:
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] Plot skipped: {e}")
        return

    df = pd.DataFrame(rows)
    if "f_cost" not in df.columns:
        df["f_cost"] = 0.0
    for c in ["seed", "gamma", "delta", "h_scale", "f_cost", "objective", "precontract_qty", "weighted_service_rate"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["f_cost"] = df["f_cost"].fillna(0.0)
    df = df[df["objective"].notna()].copy()
    if df.empty:
        return
    figdir.mkdir(parents=True, exist_ok=True)

    # Mean objective by Delta for each Gamma, one figure per (h-scale, F) pair.
    for (hs, fc), sub in df.groupby(["h_scale", "f_cost"]):
        g = sub.groupby(["gamma", "delta"], as_index=False)["objective"].mean()
        plt.figure(figsize=(8, 5))
        for gamma in sorted(g["gamma"].unique()):
            gg = g[g["gamma"] == gamma].sort_values("delta")
            plt.plot(gg["delta"], gg["objective"], marker="o", label=f"Gamma={gamma:g}")
        plt.xlabel("Auction cycle Delta (days)")
        plt.ylabel("Mean net operational surplus")
        plt.title(f"{stage}: Objective vs Delta (h scale={hs:g}, F={fc:g})")
        plt.legend()
        plt.tight_layout()
        suffix = f"h{sanitize_float(float(hs))}" + (f"_F{sanitize_float(float(fc))}" if fc else "")
        plt.savefig(figdir / f"objective_vs_delta_{suffix}.png", dpi=180)
        plt.close()

    for fc, sub in df.groupby("f_cost"):
        plt.figure(figsize=(8, 5))
        g = sub.groupby(["gamma", "delta"], as_index=False)["weighted_service_rate"].mean()
        for gamma in sorted(g["gamma"].unique()):
            gg = g[g["gamma"] == gamma].sort_values("delta")
            plt.plot(gg["delta"], gg["weighted_service_rate"], marker="o", label=f"Gamma={gamma:g}")
        plt.xlabel("Auction cycle Delta (days)")
        plt.ylabel("Mean weighted service rate")
        plt.title(f"{stage}: Service rate vs Delta" + (f" (F={fc:g})" if fc else ""))
        plt.legend()
        plt.tight_layout()
        suffix = f"_F{sanitize_float(float(fc))}" if fc else ""
        plt.savefig(figdir / f"service_rate_vs_delta{suffix}.png", dpi=180)
        plt.close()


def build_sanity_report(
    summary_path: Path,
    robust_path: Path,
    scenario_path: Path,
    capacity_path: Path,
    out_path: Path,
) -> None:
    try:
        import pandas as pd
    except Exception as e:
        out_path.write_text(f"pandas unavailable: {e}\n", encoding="utf-8")
        return

    s = pd.read_csv(summary_path) if summary_path.exists() and summary_path.stat().st_size else pd.DataFrame()
    r = pd.read_csv(robust_path) if robust_path.exists() and robust_path.stat().st_size else pd.DataFrame()
    sc = pd.read_csv(scenario_path) if scenario_path.exists() and scenario_path.stat().st_size else pd.DataFrame()

    lines: List[str] = ["5PL Model 9.2 - Stage 0 Sanity Report", "=" * 64, ""]
    if s.empty:
        lines.append("No Stage-0 results found.")
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return

    lines.append(f"Runs: {len(s)}")
    lines.append("Statuses: " + str(s["status"].value_counts().to_dict()))
    all_opt = bool((s["status"] == "OPTIMAL").all())
    lines.append(f"All optimal: {all_opt}")

    if not r.empty:
        max_dual_err = float((r["model_robust_rhs"] - r["theoretical_target"]).abs().max())
        lines.append(f"Max |model robust RHS - theoretical target|: {max_dual_err:.6g}")
        # Gamma monotonic target check for each delta/item/slot where all gammas exist.
        bad = 0
        for _, grp in r.groupby(["delta", "item", "slot"]):
            gg = grp.sort_values("gamma")
            vals = gg["theoretical_target"].to_numpy(dtype=float)
            if np.any(np.diff(vals) < -1e-7):
                bad += 1
        lines.append(f"Robust-target monotonicity violations: {bad}")

    if "objective_reconstruction_error" in s.columns:
        lines.append(
            "Max objective reconstruction error: "
            f"{float(s['objective_reconstruction_error'].abs().max()):.6g}"
        )

    for gamma, grp in s.groupby("gamma"):
        gg = grp.sort_values("delta")
        best = gg.loc[gg["objective"].idxmax()]
        lines.append(
            f"Gamma={gamma:g}: best sanity Delta={int(best['delta'])}, "
            f"objective={best['objective']:.3f}, service={best['weighted_service_rate']:.4f}"
        )

    if not sc.empty:
        piv = sc.groupby("scenario")["service_rate"].mean().sort_values(ascending=False)
        lines.append("Mean service rate by scenario: " + ", ".join(f"{k}={v:.4f}" for k, v in piv.items()))

    lines += [
        "",
        "Interpretation guide:",
        "- Robust target changes but xR does not: robust cover may still be non-binding relative to another bottleneck.",
        "- All capacity binding rates near 0: baseline may be too loose.",
        "- All capacity binding rates near 1 or infeasible: baseline may be too tight.",
        "- Delta=1 dominates every Gamma: do not force an interior optimum; first inspect whether long-cycle pooling has any modeled benefit.",
        "- Network_Disruption resembles No-shock: rho_capacity=1.25 and gamma=0.15 may be too mild; treat this as a calibration result, not a code error.",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Stage 3: recommendation and Regret
# -----------------------------------------------------------------------------


COST_TERMS = [
    "precontract_cost", "auction_cost", "weighted_slot_wait_cost", "weighted_backlog_wait_cost",
    "weighted_terminal_backlog_cost", "weighted_service_penalty",
]


def stage3_regret(results_root: Path, cfg: Optional[Mapping[str, Any]] = None) -> None:
    # Groups are (gamma, f_cost); v13 files without an f_cost column are treated as F=0.
    try:
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(f"Stage3 requires pandas/matplotlib: {e}") from e

    src = results_root / "02_stage1_baseline" / "run_summary.csv"
    if not src.exists() or src.stat().st_size == 0:
        raise RuntimeError("Stage1 run_summary.csv not found. Run stage1 first.")
    df = pd.read_csv(src)
    if "f_cost" not in df.columns:
        df["f_cost"] = 0.0
    for c in ["weighted_trade_surplus"] + COST_TERMS:
        if c not in df.columns:
            df[c] = 0.0
    num_cols = ["seed", "gamma", "delta", "h_scale", "f_cost", "objective", "weighted_service_rate",
                "precontract_qty", "weighted_final_backlog", "weighted_trade_surplus"] + COST_TERMS
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["f_cost"] = df["f_cost"].fillna(0.0)
    df = df[(df["status"] == "OPTIMAL") & df["objective"].notna() & (df["h_scale"] == 1.0)].copy()
    # If a failed combination was later rerun successfully, keep the latest optimal row.
    df = df.drop_duplicates(subset=["seed", "gamma", "delta", "h_scale", "f_cost"], keep="last")
    if cfg is not None:
        st = cfg["study"]
        expected = len(st["seeds"]) * len(st["gamma_grid"]) * len(st["delta_grid"]) * len(f_cost_grid(cfg))
    else:
        expected = 5 * 5 * 28
    if len(df) != expected:
        raise RuntimeError(
            f"Stage3 requires a complete Stage1 baseline: {expected} unique optimal rows are required, "
            f"but {len(df)} were found. Finish `python run_research.py stage1` first."
        )

    outdir = results_root / "05_stage3_regret"
    outdir.mkdir(parents=True, exist_ok=True)
    keys = ["gamma", "f_cost"]

    # Per-seed optimum
    idx = df.groupby(["seed"] + keys)["objective"].idxmax()
    per_seed = df.loc[idx, ["seed"] + keys + ["delta", "objective", "weighted_service_rate", "precontract_qty", "weighted_final_backlog"]].copy()
    per_seed = per_seed.rename(columns={"delta": "delta_star_seed", "objective": "objective_seed_opt"})
    per_seed.to_csv(outdir / "per_seed_optima.csv", index=False, encoding="utf-8-sig")

    # Common recommendation by mean objective
    mean_obj = df.groupby(keys + ["delta"], as_index=False)["objective"].mean().rename(columns={"objective": "mean_objective"})
    idx2 = mean_obj.groupby(keys)["mean_objective"].idxmax()
    rec = mean_obj.loc[idx2].copy().rename(columns={"delta": "delta_recommended"})
    rec.to_csv(outdir / "recommended_delta_by_gamma.csv", index=False, encoding="utf-8-sig")

    # Regret = seed-specific optimum minus objective at common recommended Delta.
    merged = per_seed.merge(rec[keys + ["delta_recommended", "mean_objective"]], on=keys, how="left")
    chosen = df.merge(rec[keys + ["delta_recommended"]], on=keys, how="left")
    chosen = chosen[chosen["delta"] == chosen["delta_recommended"]][["seed"] + keys + ["objective"]].rename(columns={"objective": "objective_recommended"})
    regret = merged.merge(chosen, on=["seed"] + keys, how="left")
    regret["regret"] = regret["objective_seed_opt"] - regret["objective_recommended"]
    regret["relative_regret"] = regret["regret"] / regret["objective_seed_opt"].abs().replace(0, np.nan)
    regret.to_csv(outdir / "regret_by_seed_gamma.csv", index=False, encoding="utf-8-sig")

    summary = regret.groupby(keys, as_index=False).agg(
        recommended_delta=("delta_recommended", "first"),
        mean_regret=("regret", "mean"),
        max_regret=("regret", "max"),
        mean_relative_regret=("relative_regret", "mean"),
        std_delta_star_seed=("delta_star_seed", "std"),
    )
    summary.to_csv(outdir / "regret_summary_by_gamma.csv", index=False, encoding="utf-8-sig")

    # Objective decomposition (mean over seeds) explains why a Delta wins.
    decomp = df.groupby(keys + ["delta"], as_index=False)[["objective", "weighted_trade_surplus", "weighted_service_rate"] + COST_TERMS].mean()
    decomp.to_csv(outdir / "objective_decomposition.csv", index=False, encoding="utf-8-sig")

    # Curves
    figdir = results_root / "figures" / "stage3"
    figdir.mkdir(parents=True, exist_ok=True)
    for fc, sub in mean_obj.groupby("f_cost"):
        plt.figure(figsize=(8, 5))
        for gamma in sorted(sub["gamma"].unique()):
            gg = sub[sub["gamma"] == gamma].sort_values("delta")
            plt.plot(gg["delta"], gg["mean_objective"], marker="o", label=f"Gamma={gamma:g}")
            best = rec[(rec["gamma"] == gamma) & (rec["f_cost"] == fc)]
            plt.scatter(best["delta_recommended"], best["mean_objective"], s=120, facecolors="none", edgecolors="k", zorder=5)
        plt.xlabel("Auction cycle Delta (days)")
        plt.ylabel("Mean net operational surplus over seeds")
        plt.title("Common auction-cycle selection" + (f" (F={fc:g})" if fc else ""))
        plt.legend()
        plt.tight_layout()
        suffix = f"_F{sanitize_float(float(fc))}" if fc else ""
        plt.savefig(figdir / f"mean_objective_and_recommended_delta{suffix}.png", dpi=180)
        plt.close()

    if rec["f_cost"].nunique() > 1:
        plt.figure(figsize=(7, 4.5))
        for gamma in sorted(rec["gamma"].unique()):
            gg = rec[rec["gamma"] == gamma].sort_values("f_cost")
            plt.plot(gg["f_cost"], gg["delta_recommended"], marker="o", label=f"Gamma={gamma:g}")
        plt.xlabel("Per-auction fixed cost F")
        plt.ylabel("Recommended Delta* (days)")
        plt.title("Delta* vs auction fixed cost")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figdir / "delta_star_vs_F.png", dpi=180)
        plt.close()

    # Stacked cost decomposition by Delta, one panel per F at the median Gamma.
    gam_mid = sorted(decomp["gamma"].unique())[len(decomp["gamma"].unique()) // 2]
    for fc, sub in decomp[decomp["gamma"] == gam_mid].groupby("f_cost"):
        sub = sub.sort_values("delta")
        plt.figure(figsize=(9, 5))
        bottom = np.zeros(len(sub))
        for term in COST_TERMS:
            vals = sub[term].to_numpy(dtype=float)
            if np.allclose(vals, 0.0):
                continue
            plt.bar(sub["delta"], vals, bottom=bottom, label=term.replace("weighted_", ""))
            bottom += vals
        plt.plot(sub["delta"], sub["weighted_trade_surplus"], "k--", marker=".", label="trade surplus")
        plt.plot(sub["delta"], sub["objective"], "r-", marker="o", label="objective")
        plt.xlabel("Auction cycle Delta (days)")
        plt.ylabel("Mean value over seeds")
        plt.title(f"Cost decomposition vs Delta (Gamma={gam_mid:g}" + (f", F={fc:g})" if fc else ")"))
        plt.legend(fontsize=8)
        plt.tight_layout()
        suffix = f"_F{sanitize_float(float(fc))}" if fc else ""
        plt.savefig(figdir / f"cost_decomposition_G{sanitize_float(float(gam_mid))}{suffix}.png", dpi=180)
        plt.close()

    print("\nStage3 complete")
    print(rec.to_string(index=False))
    print(f"Results: {outdir}")


# -----------------------------------------------------------------------------
# Check command and structural pre-checks
# -----------------------------------------------------------------------------


def structural_precheck(raw: RawData, cfg: Mapping[str, Any], gammas: Sequence[float], deltas: Sequence[int]) -> List[Dict[str, Any]]:
    mkt = cfg_market(cfg)
    imap = item_map(cfg)
    rows: List[Dict[str, Any]] = []
    for delta in deltas:
        agg = aggregate(raw, cfg, delta)
        for gamma in gammas:
            for t in range(agg["nslots"]):
                robust_targets: Dict[str, float] = {}
                for l, item in enumerate(mkt["items"]):
                    nominal = float(agg["d_hat"][:, l, t].sum())
                    target = nominal + theoretical_bs_buffer(agg["xi"][:, l, t], gamma)
                    robust_targets[item] = target
                    supply = float(agg["s_hat"][:, l, t].sum())
                    rows.append({
                        "delta": delta, "gamma": gamma, "slot": t + 1, "check": f"item_supply_{item}",
                        "required": target, "available": supply, "margin": supply - target,
                        "passes_necessary_check": int(supply + 1e-8 >= target),
                    })
                req_w = sum(mkt["weight"][item] * robust_targets[item] for item in mkt["items"])
                req_v = sum(mkt["volume"][item] * robust_targets[item] for item in mkt["items"])
                avail_w = float(agg["cap_w"][:, t].sum())
                avail_v = float(agg["cap_v"][:, t].sum())
                rows.append({"delta": delta, "gamma": gamma, "slot": t + 1, "check": "aggregate_weight",
                             "required": req_w, "available": avail_w, "margin": avail_w - req_w,
                             "passes_necessary_check": int(avail_w + 1e-8 >= req_w)})
                rows.append({"delta": delta, "gamma": gamma, "slot": t + 1, "check": "aggregate_volume",
                             "required": req_v, "available": avail_v, "margin": avail_v - req_v,
                             "passes_necessary_check": int(avail_v + 1e-8 >= req_v)})
    return rows


def command_check(cfg: Mapping[str, Any], results_root: Path, overwrite_data: bool) -> None:
    ensure_dirs(results_root)
    seed = 101
    raw = load_or_create_raw(cfg, results_root, seed, overwrite=overwrite_data)
    save_daily_summary(raw, cfg, results_root)
    diag = raw_diagnostics(raw, cfg)
    print(json.dumps(diag, indent=2, ensure_ascii=False))

    pre = structural_precheck(raw, cfg, [0, 4, 8], [1, 7, 14, 28])
    out = results_root / "00_raw_data" / "seed_101_structural_precheck.csv"
    write_rows(out, pre)
    failures = sum(1 for r in pre if int(r["passes_necessary_check"]) == 0)
    print(f"Structural necessary-check failures: {failures} / {len(pre)}")
    print(f"Saved raw data and diagnostics under: {results_root / '00_raw_data'}")

    try:
        gp, _ = import_gurobi()
        print("gurobipy version:", gp.gurobi.version())
        # Tiny license check.
        test = gp.Model("license_check")
        test.Params.OutputFlag = 0
        x = test.addVar(lb=0.0)
        test.setObjective(x, 1)  # maximize unbounded is fine for license initialization
        test.optimize()
        print("Gurobi initialization: OK")
    except Exception as e:
        print("Gurobi initialization: NOT READY")
        print(str(e))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="5PL two-stage robust pipeline (Model 9.2 Baseline + v14 extensions)")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to config json (config_baseline.json / config_v14.json)")
    p.add_argument("--results", type=Path, default=None, help="Results root (default: results/<study.results_subdir>)")
    p.add_argument("--overwrite-data", action="store_true", help="Regenerate cached raw data for the same seed")
    p.add_argument("--no-resume", action="store_true", help="Do not skip already-completed combinations")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--time-limit", type=float, default=None)
    p.add_argument("--solver-output", action="store_true", help="Show Gurobi log in terminal")
    p.add_argument("--write-logs", action="store_true", help="Write one Gurobi log per LP")

    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="Raw-data reproducibility and Gurobi environment check")
    sub.add_parser("stage0", help="Seed101 sanity check: Gamma {0,4,8} x Delta {1,7,14,28} at default F")
    sub.add_parser("stage1", help="Baseline main experiment: seeds x Gamma grid x F grid x Delta 1..28")
    sub.add_parser("stage2a", help="hR sensitivity pilot at default F")
    sub.add_parser("stage2b", help="hR sensitivity full Delta at default F")
    sub.add_parser("stage3", help="Recommended Delta, Regret and cost decomposition from Stage1")
    sub.add_parser("stage4", help="Heuristic policies vs exact LP (heuristics.py)")
    sub.add_parser("pricing", help="MMTR-MVCG post-processing at recommended Delta (pricing.py)")
    sub.add_parser("all", help="Run check -> stage0 -> stage1 -> stage2a -> stage3 -> stage4 -> pricing")

    one = sub.add_parser("one", help="Solve one custom LP")
    one.add_argument("--seed", type=int, default=101)
    one.add_argument("--gamma", type=float, default=4.0)
    one.add_argument("--delta", type=int, default=7)
    one.add_argument("--h-scale", type=float, default=1.0)
    one.add_argument("--f-cost", type=float, default=None, help="Per-auction fixed cost (default: extensions default)")
    one.add_argument("--export-trades", action="store_true")

    custom = sub.add_parser("custom-grid", help="Custom LP grid")
    custom.add_argument("--seeds", type=str, default="101")
    custom.add_argument("--gammas", type=str, default="0,4,8")
    custom.add_argument("--deltas", type=str, default="1,7,14,28")
    custom.add_argument("--h-scales", type=str, default="1.0")
    custom.add_argument("--f-costs", type=str, default=None)
    return p


def solver_overrides_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    d: Dict[str, Any] = {}
    if args.threads is not None:
        d["threads"] = args.threads
    if args.time_limit is not None:
        d["time_limit_sec"] = args.time_limit
    if args.solver_output:
        d["output_flag"] = 1
    if args.write_logs:
        d["write_log_files"] = True
    return d


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_json(args.config)
    if args.results is None:
        results_root = (DEFAULT_RESULTS / str(cfg["study"].get("results_subdir", "v13"))).resolve()
    else:
        results_root = args.results.resolve()
    ensure_dirs(results_root)
    resume = not args.no_resume
    solver_overrides = solver_overrides_from_args(args)
    f_default = ext_params(cfg)["f_default"]
    f_grid = f_cost_grid(cfg)
    seeds = [int(x) for x in cfg["study"]["seeds"]]
    gammas = [float(x) for x in cfg["study"]["gamma_grid"]]
    deltas = [int(x) for x in cfg["study"].get("delta_grid", list(range(1, 29)))]
    sanity_g = [float(x) for x in cfg["study"].get("sanity_gammas", [0, 4, 8])]
    sanity_d = [int(x) for x in cfg["study"].get("sanity_deltas", [1, 7, 14, 28])]
    hscales = [float(x) for x in cfg["study"]["h_scale_grid"]]
    ext_cfg = cfg.get("extensions", {})

    # Save exact config used for auditability.
    manifest = {
        "config_path": str(args.config.resolve()),
        "config_sha256": canonical_hash(cfg),
        "data_generation_fingerprint": data_generation_fingerprint(cfg),
        "command": args.command,
        "results_root": str(results_root),
        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (results_root / "last_run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    def do_stage0() -> None:
        run_grid(cfg, results_root, "stage0", [101], sanity_g, sanity_d, [1.0],
                 args.overwrite_data, resume, solver_overrides=solver_overrides, f_costs=[f_default])

    def do_stage1() -> None:
        run_grid(cfg, results_root, "stage1", seeds, gammas, deltas, [1.0],
                 args.overwrite_data, resume, solver_overrides=solver_overrides, f_costs=f_grid)

    def do_stage2(stage: str, dl: Sequence[int]) -> None:
        run_grid(cfg, results_root, stage, [101], sanity_g, dl, hscales,
                 args.overwrite_data, resume, solver_overrides=solver_overrides, f_costs=[f_default])

    def do_stage4() -> None:
        import heuristics
        g4 = [float(x) for x in ext_cfg.get("stage4_gammas", gammas)]
        heuristics.run_stage4(cfg, results_root, seeds, g4, f_grid, deltas, 1.0, resume, solver_overrides)

    def do_pricing() -> None:
        import pricing
        gp_ = [float(x) for x in ext_cfg.get("pricing_gammas", gammas)]
        pricing.run_pricing(cfg, results_root, int(ext_cfg.get("pricing_seed", 101)), gp_, f_grid, 1.0, solver_overrides)

    if args.command == "check":
        command_check(cfg, results_root, args.overwrite_data)
        return 0
    if args.command == "stage0":
        do_stage0()
        return 0
    if args.command == "stage1":
        do_stage1()
        return 0
    if args.command == "stage2a":
        do_stage2("stage2a", sanity_d)
        return 0
    if args.command == "stage2b":
        do_stage2("stage2b", deltas)
        return 0
    if args.command == "stage3":
        stage3_regret(results_root, cfg)
        return 0
    if args.command == "stage4":
        do_stage4()
        return 0
    if args.command == "pricing":
        do_pricing()
        return 0

    if args.command == "one":
        raw = load_or_create_raw(cfg, results_root, args.seed, overwrite=args.overwrite_data)
        fc = f_default if args.f_cost is None else float(args.f_cost)
        summary, robust, scen, cap, trades = solve_lp(
            cfg, raw, results_root, "one", args.seed, args.gamma, args.delta, args.h_scale, f_cost=fc,
            export_positive_trades=args.export_trades, solver_overrides=solver_overrides,
        )
        one_dir = results_root / "one"
        one_dir.mkdir(parents=True, exist_ok=True)
        stem = f"seed{args.seed}_G{sanitize_float(args.gamma)}_D{args.delta}_h{sanitize_float(args.h_scale)}_F{sanitize_float(fc)}"
        write_rows(one_dir / f"{stem}_summary.csv", [summary])
        write_rows(one_dir / f"{stem}_robust.csv", robust)
        write_rows(one_dir / f"{stem}_scenarios.csv", scen)
        write_rows(one_dir / f"{stem}_capacity.csv", cap)
        if args.export_trades:
            write_rows(one_dir / f"{stem}_positive_trades.csv", trades)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    if args.command == "custom-grid":
        cseeds = parse_num_list(args.seeds, int) or [101]
        cgammas = parse_num_list(args.gammas, float) or [0.0, 4.0, 8.0]
        cdeltas = parse_num_list(args.deltas, int) or [1, 7, 14, 28]
        chs = parse_num_list(args.h_scales, float) or [1.0]
        cfs = parse_num_list(args.f_costs, float) or [f_default]
        run_grid(cfg, results_root, "stage0", cseeds, cgammas, cdeltas, chs,
                 args.overwrite_data, resume, solver_overrides=solver_overrides, f_costs=cfs)
        return 0

    if args.command == "all":
        command_check(cfg, results_root, args.overwrite_data)
        do_stage0()
        do_stage1()
        do_stage2("stage2a", sanity_d)
        stage3_regret(results_root, cfg)
        do_stage4()
        do_pricing()
        return 0

    raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
