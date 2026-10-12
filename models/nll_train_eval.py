"""p10 — 값과 흔들림을 함께 예측하는 모델 (가우시안 NLL). train_eval.py는 고치지 않고 불러다 쓴다.

출력 20 = μ 10 + σ² 10 (softplus + 1e-6), 손실 GaussianNLLLoss, 조기 종료는 검증 NLL.
판정은 지금과 같다: β = exp(−dt/τ) 고정 평활 → (비행별이면 기준선으로 나눔) → T = 검증 μ + 3.291σ.
점수 두 가지를 같은 모델에서 계산해 run 폴더를 따로 만든다 (eval_variants.py가 그대로 읽는 꼴):
  <out>/<시각>_<tag>_z  : dₜ = ‖(y − μ)/σ‖₂   (주 점수)
  <out>/<시각>_<tag>_mu : dₜ = ‖y − μ‖₂       (진단용)
사용: python nll_train_eval.py --config <p01 config json> --out <폴더> --tag <이름> [--seeds 0,1,2,3,4]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_eval as te  # noqa: E402
from model import TransformerPredictor  # noqa: E402

N_STATE = 10


def split(out):
    return out[:, :N_STATE], F.softplus(out[:, N_STATE:]) + 1e-6


def train(data, c, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr = TensorDataset(torch.from_numpy(data["normal"]["X"]), torch.from_numpy(data["normal"]["Y"]))
    va = TensorDataset(torch.from_numpy(data["valid"]["X"]), torch.from_numpy(data["valid"]["Y"]))
    tr_dl = DataLoader(tr, batch_size=c["batch"], shuffle=True, drop_last=True)
    va_dl = DataLoader(va, batch_size=c["batch"], shuffle=False)
    model = TransformerPredictor(window_size=c["L"], d_model=c["d_model"], nhead=c["nhead"], num_layers=c["layers"],
                                 n_in=c["n_in"], n_out=2 * N_STATE, norm=c["norm"], pool=c["pool"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=c["lr"], betas=(0.9, 0.98), eps=1e-9)
    crit = nn.GaussianNLLLoss(eps=1e-6)
    best, best_state, wait, hist = float("inf"), None, 0, []
    for ep in range(c["epochs"]):
        model.train()
        tl = 0.0
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            mu, var = split(model(xb))
            loss = crit(mu, yb, var)
            loss.backward()
            opt.step()
            tl += loss.item() * xb.size(0)
        tl /= len(tr_dl.dataset)
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for xb, yb in va_dl:
                xb, yb = xb.to(device), yb.to(device)
                mu, var = split(model(xb))
                vl += crit(mu, yb, var).item() * xb.size(0)
        vl /= len(va_dl.dataset)
        hist.append((ep, tl, vl))
        if vl < best:
            best, wait = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= c["patience"]:
                break
    model.load_state_dict(best_state)
    return model, best, hist


@torch.no_grad()
def predict(model, X, device, batch=4096):
    model.eval()
    mus, vs = [], []
    for i in range(0, len(X), batch):
        mu, var = split(model(torch.from_numpy(X[i:i + batch]).to(device)))
        mus.append(mu.cpu().numpy())
        vs.append(var.cpu().numpy())
    return np.concatenate(mus), np.concatenate(vs)


def judge(scores, data, c, beta):
    """평활 → (비행별) 기준선 → T. scores: 세트별 (pred, Y) — ewma_residual이 ‖Y − pred‖을 평활한다."""
    res = {s: te.ewma_residual(p, y, data[s]["seq_length"], c["L"], beta) for s, (p, y) in scores.items()}
    if c["threshold_mode"] == "per-flight":
        res = {s: te.normalize_per_flight(r, data[s]["seq_length"], c["L"], c["baseline_steps"]) for s, r in res.items()}
    vr = res["valid"]
    return res, float(vr.mean() + te.SIGMA * vr.std())


def write(run_dir, seed, res, T, data, c, extra):
    sd = os.path.join(run_dir, f"seed{seed}")
    os.makedirs(sd, exist_ok=True)
    parts = []
    for s in te.SETS:
        sl = data[s]["seq_length"]
        for k, (r, f) in enumerate(zip(te.split_by_scenario(res[s], sl, c["L"]),
                                       te.split_by_scenario(data[s]["fault"], sl, c["L"]))):
            j = np.arange(len(r))
            parts.append(pd.DataFrame(dict(set=s, scenario=k, step=j, t_s=np.round(te.DT * j, 2),
                                           residual=r, threshold=T, fault=f.astype(int))))
    pd.concat(parts).to_csv(os.path.join(sd, "residuals.csv"), index=False)
    m = te.evaluate(res, data, c["L"], T)
    auroc, auprc, pos = te.ranking_metrics(res, data, te.FAULT_SETS)
    return dict(seed=seed, auroc=auroc, auprc=auprc, pos_rate=pos, threshold=T, scen_TP=m["scen_TP"],
                scen_FP=m["scen_FP"], scen_FN=m["scen_FN"], scen_F1=m["scen_F1"], delay_mean_s=m["delay_mean"],
                n_detected=m["n_detected"], n_scenarios=m["n_scenarios"], **extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", default="")
    ap.add_argument("--tau", type=float, default=1.3)
    a = ap.parse_args()
    c = json.load(open(a.config))
    te.SETS = c["sets"].split(",")
    te.FAULT_SETS = c["fault_sets"].split(",")
    te.DT = c["dt"]
    beta = float(np.exp(-c["dt"] / a.tau))
    seeds = [int(s) for s in (a.seeds or c["seeds"]).split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = te.load_npz(c["npz"], c["L"])
    stamp = time.strftime("%m%d-%H%M%S")
    dirs = {k: os.path.join(a.out, f"{stamp}_{a.tag}_{k}") for k in ("z", "mu")}
    for k, d in dirs.items():
        os.makedirs(d, exist_ok=True)
        json.dump(c | dict(tag=f"{a.tag}_{k}", model="gaussian NLL (mu+var)", score=k, beta_rule=f"exp(-dt/tau), tau={a.tau}",
                           beta=beta, device=str(device), source_config=a.config),
                  open(os.path.join(d, "config.json"), "w"), indent=1, ensure_ascii=False)
    print(f"device={device} β={beta:.4f} → {dirs['z']} / _mu", flush=True)
    rows = {k: [] for k in dirs}
    for seed in seeds:
        t0 = time.time()
        model, vnll, hist = train(data, c, seed, device)
        torch.save(model.state_dict(), os.path.join(dirs["z"], f"model_seed{seed}.pt"))
        pd.DataFrame(hist, columns=["epoch", "train_nll", "val_nll"]).to_csv(
            os.path.join(dirs["z"], f"train_history_seed{seed}.csv"), index=False)
        pred = {s: predict(model, data[s]["X"], device) for s in te.SETS}
        sc = {"z": {s: (np.zeros_like(mu), (data[s]["Y"] - mu) / np.sqrt(v)) for s, (mu, v) in pred.items()},
              "mu": {s: (mu, data[s]["Y"]) for s, (mu, v) in pred.items()}}
        extra = dict(val_nll=vnll, epochs_run=len(hist), train_s=time.time() - t0)
        for k in dirs:
            res, T = judge(sc[k], data, c, beta)
            rows[k].append(write(dirs[k], seed, res, T, data, c, extra))
            r = rows[k][-1]
            print(f"seed {seed} [{k:2s}] val_nll {vnll:.4f} ep {len(hist)} | AUROC {r['auroc']:.4f} AUPRC {r['auprc']:.4f} "
                  f"F1 {r['scen_F1']:.3f} TP {r['scen_TP']} FP {r['scen_FP']} / {r['n_scenarios']}", flush=True)
    for k, d in dirs.items():
        pd.DataFrame(rows[k]).to_csv(os.path.join(d, "metrics.csv"), index=False)
    print("완료", dirs["z"], flush=True)


if __name__ == "__main__":
    main()
