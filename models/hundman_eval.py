"""판정 단계 — Hundman 2018 동적 임계값 + 가지치기. 저장된 잔차(residuals.csv)로 재평가, 재학습 없음.

R0 기준(eₛ ≥ T) / R1 가지치기만 / R2 동적 임계값 + 가지치기 / R3 R2 + 하한 T. 규칙은 models/README.md.
파라미터는 논문 값 고정: z ∈ {2.5, 3.0, …, 10.0}, p = 0.13. 비행 하나 전체를 한 묶음으로 판정한다.
평가는 eval_variants.py와 같은 V0~V3. AUROC·AUPRC는 원래 점수로 계산(판정 규칙과 무관).
경보는 0/1 점수로 바꿔 임계 0.5로 train_eval.evaluate에 넘긴다(탐지·오경보·지연 계산 규칙은 원본 그대로).
사용: python hundman_eval.py --run <run_tau 폴더> --dt 0.1 --exclude <json> --npz <npz> --out <폴더> --tag <이름>
산출: <out>/<tag>.csv (규칙 × V0~V3 × seed)
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_eval as te  # noqa: E402
from eval_variants import VARIANTS, build, load_seed, resolve_exclude  # noqa: E402

Z = np.arange(2.5, 10.01, 0.5)
P = 0.13
RULES = ["R0_기준", "R1_가지치기", "R2_동적+가지치기", "R3_동적+가지치기+하한"]


def runs(mask):
    """연속 True 구간의 (시작, 끝+1) 목록."""
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def prune(e, mask, p=P):
    """논문 3.3절: 이상 구간 최댓값을 내림차순으로 놓고 끝에 구간 밖 최댓값을 붙여, 감소율이 p를 넘는 마지막 지점까지만 남긴다."""
    seqs = runs(mask)
    if not seqs:
        return mask
    mx = np.array([e[a:b].max() for a, b in seqs])
    order = np.argsort(-mx, kind="mergesort")
    rest = e[~mask]
    emax = np.concatenate([mx[order], [rest.max() if rest.size else 0.0]])
    dec = (emax[:-1] - emax[1:]) / np.where(emax[:-1] > 0, emax[:-1], 1.0)   # d⁽ⁱ⁾, i = 1 … |Eseq|
    over = np.flatnonzero(dec > p)
    keep_n = over[-1] + 1 if over.size else 0                               # 감소율이 p를 넘는 마지막 지점 앞까지
    out = np.zeros_like(mask)
    for j in order[:keep_n]:
        a, b = seqs[j]
        out[a:b] = True
    return out


def dynamic_eps(e):
    """논문 3.2절: argmax [Δμ/μ + Δσ/σ] / (|eₐ| + |Eseq|²). 이상을 만드는 z가 없으면 무한대(경보 없음)."""
    mu, sd = e.mean(), e.std()
    best, eps = -np.inf, np.inf
    if sd <= 0 or mu <= 0:
        return eps
    for z in Z:
        t = mu + z * sd
        below, above = e[e < t], e >= t
        n_a = int(above.sum())
        if n_a == 0 or below.size == 0:
            continue
        n_seq = len(runs(above))
        s = ((mu - below.mean()) / mu + (sd - below.std()) / sd) / (n_a + n_seq ** 2)
        if s > best:
            best, eps = s, t
    return eps


def alarms(e, T, rule):
    if rule == "R0_기준":
        return e >= T
    if rule == "R1_가지치기":
        return prune(e, e >= T)
    a = prune(e, e >= dynamic_eps(e))
    return a & (e >= T) if rule == "R3_동적+가지치기+하한" else a


ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--dt", type=float, required=True)
ap.add_argument("--exclude", default="")
ap.add_argument("--npz", default="")
ap.add_argument("--out", required=True)
ap.add_argument("--tag", required=True)
args = ap.parse_args()
cfg = json.load(open(os.path.join(args.run, "config.json")))
L = cfg["L"]
fault_sets = cfg["fault_sets"].split(",")
te.FAULT_SETS, te.DT = fault_sets, args.dt
drop_idx, _ = resolve_exclude(args.exclude, args.npz)
seeds = sorted(int(d[4:]) for d in os.listdir(args.run) if d.startswith("seed"))

rows = []
for s in seeds:
    per, T = load_seed(os.path.join(args.run, f"seed{s}", "residuals.csv"), L)
    for rule in RULES:
        pa = {st: [(alarms(r, T, rule).astype(float), f) for r, f in v] for st, v in per.items()}
        vfrac = float(np.mean([a.any() for a, _ in pa["valid"]]))
        for var in VARIANTS:
            v2 = var in ("V2_목록제외", "V3_둘다")
            kw = dict(drop_post=var in ("V1_결함후제외", "V3_둘다"), drop_idx=drop_idx if v2 else set(), drop_nofault=v2)
            res, data = build(pa, fault_sets, L, **kw)
            res0, _ = build(per, fault_sets, L, **kw)
            m = te.evaluate(res, data, L, 0.5)
            auroc, auprc, _ = te.ranking_metrics(res0, data, fault_sets)
            rows.append(dict(rule=rule, seed=s, variant=var, valid_alarm_frac=vfrac, auroc=auroc, auprc=auprc,
                             scen_F1=m["scen_F1"], scen_TP=m["scen_TP"], scen_FP=m["scen_FP"],
                             n=m["n_scenarios"], delay_s=m["delay_mean"]))
df = pd.DataFrame(rows)
os.makedirs(args.out, exist_ok=True)
df.to_csv(os.path.join(args.out, f"{args.tag}.csv"), index=False)
g = df.groupby(["rule", "variant"], sort=False)
mean, std = g.mean(numeric_only=True), g.std(numeric_only=True)
print(f"== {args.tag}")
for rule in RULES:
    for var in ("V0_원래", "V3_둘다"):
        r, sd = mean.loc[(rule, var)], std.loc[(rule, var)]
        print(f"  {rule:22s} {var:8s} F1 {r.scen_F1:.3f}±{sd.scen_F1:.3f} | 탐지 {r.scen_TP:.1f}/{r.n:.0f} | "
              f"오경보 {r.scen_FP:.1f} | 지연 {r.delay_s:.1f} s | 검증 경보 {r.valid_alarm_frac:.2f} | AUROC {r.auroc:.3f}")
