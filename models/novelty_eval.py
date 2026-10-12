"""p08 — 기존 잔차 점수 + 입력 지령 이탈 점수를 max로 합친다. 재학습 없음.
--run: β 고정 평가가 끝난 run(residuals.csv에 정규화된 잔차·임계 T_r). --npz: 그 run의 npz. --n-state: 상태(예측 대상) 채널 수.
산출: <out>/ config.json, metrics.csv, seed*/residuals.csv (residual = s, threshold = 1) → eval_variants.py
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_eval as te  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--npz", required=True)
ap.add_argument("--n-state", type=int, required=True)
ap.add_argument("--dt", type=float, required=True)
ap.add_argument("--tau", type=float, default=1.3)
ap.add_argument("--out", required=True)
args = ap.parse_args()
cfg = json.load(open(os.path.join(args.run, "config.json")))
L = cfg["L"]
sets, fsets = cfg["sets"].split(","), cfg["fault_sets"].split(",")
te.SETS, te.FAULT_SETS, te.DT = sets, fsets, args.dt
data = te.load_npz(args.npz, L)
beta = float(np.exp(-args.dt / args.tau))
nov = {}
for s in sets:
    raw = np.abs(data[s]["X"][:, -1, args.n_state:]).max(1)
    out, i = np.empty_like(raw), 0
    for n in data[s]["seq_length"]:
        for j in range(int(n) - L):
            out[i] = raw[i] if j == 0 else beta * out[i - 1] + (1 - beta) * raw[i]
            i += 1
    nov[s] = out
Tn = float(nov["valid"].mean() + te.SIGMA * nov["valid"].std())
os.makedirs(args.out, exist_ok=True)
rows = []
for sd in sorted(d for d in os.listdir(args.run) if d.startswith("seed")):
    d = pd.read_csv(os.path.join(args.run, sd, "residuals.csv")).sort_values(["set", "scenario", "step"], kind="stable")
    Tr = float(d.threshold.iloc[0])
    sc = {}
    for s in sets:
        r = d[d.set == s].residual.to_numpy()
        assert len(r) == len(nov[s]), (s, len(r), len(nov[s]))
        sc[s] = np.maximum(r / Tr, nov[s] / Tn)
    m = te.evaluate(sc, data, L, 1.0)
    auroc, auprc, pos = te.ranking_metrics(sc, data, fsets)
    rows.append(dict(seed=int(sd[4:]), threshold=1.0, T_r=Tr, T_n=Tn, auroc=auroc, auprc=auprc, pos_rate=pos, scen_TP=m["scen_TP"],
                     scen_FP=m["scen_FP"], scen_FN=m["scen_FN"], scen_F1=m["scen_F1"], delay_mean_s=m["delay_mean"], n_scenarios=m["n_scenarios"]))
    od = os.path.join(args.out, sd)
    os.makedirs(od, exist_ok=True)
    recs = []
    for s in sets:
        for k, (r, f) in enumerate(zip(te.split_by_scenario(sc[s], data[s]["seq_length"], L),
                                       te.split_by_scenario(data[s]["fault"], data[s]["seq_length"], L))):
            recs.append(pd.DataFrame(dict(set=s, scenario=k, step=np.arange(len(r)), t_s=np.arange(len(r)) * args.dt,
                                          residual=r, threshold=1.0, fault=f.astype(int))))
    pd.concat(recs).to_csv(os.path.join(od, "residuals.csv"), index=False)
    print(f"  {sd}: AUROC {auroc:.4f} F1 {m['scen_F1']:.3f} TP {m['scen_TP']} FP {m['scen_FP']}")
pd.DataFrame(rows).to_csv(os.path.join(args.out, "metrics.csv"), index=False)
json.dump(dict(cfg, novelty="max |z| of command inputs, EWMA", T_n=Tn, derived_from=args.run), open(os.path.join(args.out, "config.json"), "w"),
          ensure_ascii=False, indent=1)
