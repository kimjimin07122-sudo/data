"""트랙 C — 자체 Transformer 예측기 학습 · 평가

판정 규칙은 park 3장(총상태 스칼라 residual) 하나로 고정한다. REPRODUCE.md 6절 참고.
  residual  r[j] = b*r[j-1] + (1-b)*||Y-pred||   (시나리오마다 j==0에서 리셋)
  beta      0.700 + 0.001*i, i=0..270 (271점)  →  |T - max(valid_r)| 최소인 b
  threshold T = mean(valid_r) + 3.291*std(valid_r)
  지연      0.25 * (첫 초과 인덱스 - 결함 시작 인덱스), 탐지된 시나리오만 (개수 함께 보고)

park 노트북 셀 31~48(채널별 residual)은 논문의 reference [21] 재현이므로 쓰지 않는다.

사용법:
  python transformer/train_eval.py --seeds 0,1,2,3,4
"""

import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import TransformerPredictor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 기본은 ALFA. --sets / --fault-sets 로 다른 데이터셋(UAV-SEAD 등)에도 쓴다.
SETS = ["normal", "valid", "engine", "elevator", "rudder"]
FAULT_SETS = ["engine", "elevator", "rudder"]
DT = 0.25          # 한 스텝의 초. ALFA 기본 4 Hz. --dt 로 바꾼다.
                   # UAV-SEAD는 5 Hz(0.2), ArduFault는 10 Hz(0.1)라 반드시 지정해야
                   # 지연·t_s가 맞는다. 안 바꾸면 지연이 주기 비율만큼 부풀려진다.
SIGMA = 3.291      # 한쪽 꼬리 99.95%


# ---------------------------------------------------------------- 데이터

def load_npz(path, window):
    d = np.load(path)
    data = {}
    for s in SETS:
        X, Y = d[f"{s}_X"], d[f"{s}_Y"]
        lab, sl = d[f"{s}_label"], d[f"{s}_seq_length"]
        n_res = int((sl - window).sum())
        assert n_res == X.shape[0], f"{s}: sum(seq_length-L)={n_res} != X={X.shape[0]}"
        data[s] = dict(X=X, Y=Y, label=lab, seq_length=sl,
                       fault=lab[:, 1:].sum(axis=1) > 0)
    return data


def split_by_scenario(arr, seq_length, window):
    """시나리오별로 자른다. 시나리오당 길이는 seq_length[i] - window."""
    out, idx = [], 0
    for n in seq_length:
        k = int(n) - window
        out.append(arr[idx:idx + k])
        idx += k
    return out


# ---------------------------------------------------------------- 학습

def train(data, args, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    tr = TensorDataset(torch.from_numpy(data["normal"]["X"]),
                       torch.from_numpy(data["normal"]["Y"]))
    va = TensorDataset(torch.from_numpy(data["valid"]["X"]),
                       torch.from_numpy(data["valid"]["Y"]))
    tr_dl = DataLoader(tr, batch_size=args.batch, shuffle=True, drop_last=True)
    va_dl = DataLoader(va, batch_size=args.batch, shuffle=False)

    model = TransformerPredictor(window_size=args.L, d_model=args.d_model,
                                 nhead=args.nhead, num_layers=args.layers,
                                 n_in=args.n_in, n_out=args.n_out,
                                 norm=args.norm, pool=args.pool).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    crit = nn.MSELoss()

    best, best_state, wait, history = float("inf"), None, 0, []
    for ep in range(args.epochs):
        model.train()
        tr_loss = 0.0
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(tr_dl.dataset)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in va_dl:
                xb, yb = xb.to(device), yb.to(device)
                va_loss += crit(model(xb), yb).item() * xb.size(0)
        va_loss /= len(va_dl.dataset)
        history.append((ep, tr_loss, va_loss))

        if va_loss < best:
            best, wait = va_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                break

    model.load_state_dict(best_state)
    return model, best, history


@torch.no_grad()
def predict(model, X, device, batch=512):
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).to(device)
        out.append(model(xb).cpu().numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------- 판정 규칙

def ewma_residual(pred, Y, seq_length, window, beta):
    """park 셀 14 그대로. 시나리오 경계에서 리셋."""
    res = np.zeros(pred.shape[0])
    idx = 0
    for n in seq_length:
        for j in range(int(n) - window):
            d = float(np.linalg.norm(Y[idx] - pred[idx]))
            res[idx] = d if j == 0 else beta * res[idx - 1] + (1 - beta) * d
            idx += 1
    return res


def calibrate(valid_pred, data, window, betas):
    """park 셀 16/18. 셀 16과 달리 예측은 루프 밖에서 한 번만 한다."""
    vY, vsl = data["valid"]["Y"], data["valid"]["seq_length"]
    rows = []
    for b in betas:
        r = ewma_residual(valid_pred, vY, vsl, window, b)
        T = r.mean() + SIGMA * r.std()
        rows.append((float(b), float(T), float(abs(T - r.max()))))
    best = min(rows, key=lambda t: t[2])
    return best[0], best[1], rows


def per_flight_baseline(residuals, seq_length, window, n_base):
    """비행마다 앞 n_base 스텝의 중앙값을 그 비행의 기준선으로 쓴다.

    전역 임계는 비행별 정상 residual이 12배까지 벌어지는 데이터에서 무너진다
    (ALFA는 1.9배). 기준선으로 나누면 '평소보다 몇 배 튀었나'를 보게 된다.
    기준선은 비행 앞부분만 쓴다 — 전체 중앙값을 쓰면 결함 구간이 기준선에 섞여
    스스로 탐지를 방해하고, 라벨을 보고 자르면 평가로 반칙이다.
    """
    out, idx = [], 0
    for n in seq_length:
        k = int(n) - window
        r = residuals[idx:idx + k]
        b = float(np.median(r[:min(n_base, k)]))
        out.append(max(b, 1e-6))
        idx += k
    return np.array(out)


def normalize_per_flight(residuals, seq_length, window, n_base):
    base = per_flight_baseline(residuals, seq_length, window, n_base)
    out, idx = np.empty_like(residuals), 0
    for b, n in zip(base, seq_length):
        k = int(n) - window
        out[idx:idx + k] = residuals[idx:idx + k] / b
        idx += k
    return out


def ranking_metrics(residuals, data, fault_sets):
    """AUROC / AUPRC — 임계값과 무관한 지표.

    beta·threshold 규칙이 데이터에 따라 퇴화할 수 있으므로(UAV-SEAD에서는 |T-max|가
    단조 감소해 격자 끝이 뽑혔다), residual 점수 자체의 변별력을 따로 본다.
    AeroTSBoost 논문(arXiv 2605.25639)이 AUPRC를 주 지표로 쓰므로 비교도 가능하다.
    """
    y = np.concatenate([data[s]["fault"] for s in fault_sets])
    x = np.concatenate([residuals[s] for s in fault_sets])
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == len(y):
        return float("nan"), float("nan"), 0.0
    order = np.argsort(-x, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(~y_sorted)
    tpr = tp / n_pos
    fpr = fp / (len(y) - n_pos)
    auroc = float(np.trapz(tpr, fpr))
    precision = tp / np.arange(1, len(y) + 1)
    auprc = float(np.sum(np.diff(np.concatenate([[0.0], tpr])) * precision))
    return auroc, auprc, float(y.mean())


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def evaluate(residuals, data, window, T):
    """셀 26 (A)(B) + 셀 50/51의 지연. 전부 총상태 residual 하나로."""
    # (A) 타임스텝 단위
    ts = {}
    TP = FP = FN = 0
    for s in FAULT_SETS:
        yt, yp = data[s]["fault"], residuals[s] >= T
        tp, fp, fn = int((yp & yt).sum()), int((yp & ~yt).sum()), int((~yp & yt).sum())
        ts[s] = prf(tp, fp, fn)
        TP, FP, FN = TP + tp, FP + fp, FN + fn
    ts["전체"] = prf(TP, FP, FN)

    # (B) 시나리오 단위 + 지연
    scen, STP = [], 0
    SFP = SFN = 0
    for s in FAULT_SETS:
        res_i = split_by_scenario(residuals[s], data[s]["seq_length"], window)
        lab_i = split_by_scenario(data[s]["fault"], data[s]["seq_length"], window)
        for k, (r, yt) in enumerate(zip(res_i, lab_i)):
            over = r >= T
            start = int(np.argmax(yt)) if yt.any() else -1
            det = bool(over[yt].any())
            fa = bool(over[~yt].any())
            if det:
                rel = int(np.argmax(over[start:]))
                delay = DT * rel
                d_idx = start + rel
            else:
                delay, d_idx = float("nan"), -1
            STP += det
            SFN += (not det)
            SFP += fa
            scen.append(dict(set=s, idx=k, n_steps=len(r), fault_start_idx=start,
                             detect_idx=d_idx, delay_s=delay,
                             detected=int(det), false_alarm=int(fa),
                             max_residual=float(r.max())))
    sp, sr, sf1 = prf(STP, SFP, SFN)

    delays = [d["delay_s"] for d in scen if d["detected"]]
    n_total = len(scen)
    return dict(timestep=ts,
                scen_TP=STP, scen_FP=SFP, scen_FN=SFN,
                scen_P=sp, scen_R=sr, scen_F1=sf1,
                delay_mean=float(np.mean(delays)) if delays else float("nan"),
                delay_max=float(np.max(delays)) if delays else float("nan"),
                n_detected=len(delays), n_scenarios=n_total,
                scenarios=scen)


# ---------------------------------------------------------------- 실행

def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def run_seed(data, args, seed, device, outdir):
    t0 = time.time()
    model, val_loss, history = train(data, args, seed, device)

    preds = {s: predict(model, data[s]["X"], device) for s in SETS}
    betas = np.array([0.700 + 0.001 * i for i in range(271)])
    beta, T, grid = calibrate(preds["valid"], data, args.L, betas)

    residuals = {s: ewma_residual(preds[s], data[s]["Y"], data[s]["seq_length"], args.L, beta)
                 for s in SETS}

    if args.threshold_mode == "per-flight":
        # 비행별 기준선으로 나눈 뒤, 같은 규칙(mean + 3.291*std)을 검증셋에서 다시 잡는다.
        residuals = {s: normalize_per_flight(residuals[s], data[s]["seq_length"],
                                             args.L, args.baseline_steps)
                     for s in SETS}
        vr = residuals["valid"]
        T = float(vr.mean() + SIGMA * vr.std())

    m = evaluate(residuals, data, args.L, T)
    auroc, auprc, pos_rate = ranking_metrics(residuals, data, FAULT_SETS)
    m.update(auroc=auroc, auprc=auprc, pos_rate=pos_rate)
    m.update(seed=seed, beta=beta, threshold=T, val_mse=val_loss,
             epochs_run=len(history), train_s=time.time() - t0,
             n_params=sum(p.numel() for p in model.parameters()))

    sd = os.path.join(outdir, f"seed{seed}")
    os.makedirs(sd, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(sd, "model.pt"))
    write_csv(os.path.join(sd, "beta_search.csv"),
              [dict(beta=b, threshold=t, abs_T_minus_max=e) for b, t, e in grid],
              ["beta", "threshold", "abs_T_minus_max"])
    write_csv(os.path.join(sd, "train_history.csv"),
              [dict(epoch=e, train_mse=a, val_mse=v) for e, a, v in history],
              ["epoch", "train_mse", "val_mse"])
    write_csv(os.path.join(sd, "scenarios.csv"), m["scenarios"],
              ["set", "idx", "n_steps", "fault_start_idx", "detect_idx", "delay_s",
               "detected", "false_alarm", "max_residual"])
    rows = []
    for s in SETS:
        sl = data[s]["seq_length"]
        for k, (r, yt) in enumerate(zip(split_by_scenario(residuals[s], sl, args.L),
                                        split_by_scenario(data[s]["fault"], sl, args.L))):
            for j in range(len(r)):
                rows.append(dict(set=s, scenario=k, step=j, t_s=round(DT * j, 2),
                                 residual=float(r[j]), threshold=T, fault=int(yt[j])))
    write_csv(os.path.join(sd, "residuals.csv"), rows,
              ["set", "scenario", "step", "t_s", "residual", "threshold", "fault"])
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, default=24, help="window (export/alfa_L{L}.npz)")
    p.add_argument("--data", default=os.path.join(REPO, "transformer", "data"))
    p.add_argument("--npz", default="", help="npz 전체 경로. 비우면 <data>/alfa_L{L}.npz")
    p.add_argument("--sets", default="", help="쉼표 구분. 비우면 ALFA 기본")
    p.add_argument("--fault-sets", default="", help="쉼표 구분. 비우면 ALFA 기본")
    p.add_argument("--n_in", type=int, default=17)
    p.add_argument("--n_out", type=int, default=10)
    p.add_argument("--threshold-mode", default="global", choices=["global", "per-flight"],
                   help="global=park 원 규칙(전역 임계). per-flight=비행별 기준선으로 "
                        "residual을 나눈 뒤 임계. 이종/다양한 비행에서 전역 임계가 무너질 때.")
    p.add_argument("--baseline-steps", type=int, default=100,
                   help="per-flight 기준선에 쓸 비행 앞부분 스텝 수 (5 Hz면 100=20초). "
                        "짧으면 이륙 과도구간이라 대표성이 없다 — 20스텝에서는 정규화 후 "
                        "산포가 오히려 12배->30배로 커졌고, 100스텝에서 6.8배로 줄었다.")
    p.add_argument("--out", default=os.path.join(REPO, "results", "trackC"))
    p.add_argument("--tag", default="")
    p.add_argument("--seeds", default="0", help="쉼표 구분")
    p.add_argument("--d_model", type=int, default=32)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--norm", default="batch", choices=["batch", "layer"])
    p.add_argument("--pool", default="gap", choices=["gap", "last"])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch", type=int, default=192)
    p.add_argument("--lr", type=float, default=0.0012993)
    p.add_argument("--dt", type=float, default=0.25,
                   help="한 스텝의 초. 지연과 residuals.csv의 t_s에 쓴다. "
                        "ALFA 0.25 / UAV-SEAD 0.2 / ArduFault 0.1")
    args = p.parse_args()

    global SETS, FAULT_SETS, DT
    if args.sets:
        SETS = [x.strip() for x in args.sets.split(",")]
    if args.fault_sets:
        FAULT_SETS = [x.strip() for x in args.fault_sets.split(",")]
    DT = args.dt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    npz = args.npz or os.path.join(args.data, f"alfa_L{args.L}.npz")
    data = load_npz(npz, args.L)

    tag = args.tag or f"L{args.L}_d{args.d_model}_h{args.nhead}_l{args.layers}_{args.norm}_{args.pool}"
    outdir = os.path.join(args.out, f"{time.strftime('%m%d-%H%M%S')}_{tag}")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(vars(args) | {"device": str(device)}, f, indent=2, ensure_ascii=False)

    print(f"device={device}  out={outdir}")
    results = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        m = run_seed(data, args, seed, device, outdir)
        results.append(m)
        print(f"seed {seed}: params={m['n_params']} val_mse={m['val_mse']:.5f} "
              f"beta={m['beta']:.4f} T={m['threshold']:.4f} | "
              f"타임스텝 F1={m['timestep']['전체'][2]:.3f} | "
              f"시나리오 TP={m['scen_TP']} FP={m['scen_FP']} FN={m['scen_FN']} "
              f"F1={m['scen_F1']:.3f} | 지연 {m['delay_mean']:.2f}s "
              f"(탐지 {m['n_detected']}/{m['n_scenarios']}) | "
              f"AUROC={m['auroc']:.4f} AUPRC={m['auprc']:.4f}")

    fields = ["seed", "n_params", "val_mse", "epochs_run", "beta", "threshold",
              "auroc", "auprc", "pos_rate",
              "ts_P", "ts_R", "ts_F1", "scen_TP", "scen_FP", "scen_FN",
              "scen_P", "scen_R", "scen_F1", "delay_mean_s", "delay_max_s",
              "n_detected", "n_scenarios", "train_s"]
    rows = [dict(seed=m["seed"], auroc=m["auroc"], auprc=m["auprc"], pos_rate=m["pos_rate"],
                 n_params=m["n_params"], val_mse=m["val_mse"],
                 epochs_run=m["epochs_run"], beta=m["beta"], threshold=m["threshold"],
                 ts_P=m["timestep"]["전체"][0], ts_R=m["timestep"]["전체"][1],
                 ts_F1=m["timestep"]["전체"][2], scen_TP=m["scen_TP"], scen_FP=m["scen_FP"],
                 scen_FN=m["scen_FN"], scen_P=m["scen_P"], scen_R=m["scen_R"],
                 scen_F1=m["scen_F1"], delay_mean_s=m["delay_mean"],
                 delay_max_s=m["delay_max"], n_detected=m["n_detected"],
                 n_scenarios=m["n_scenarios"], train_s=m["train_s"]) for m in results]
    write_csv(os.path.join(outdir, "metrics.csv"), rows, fields)

    f1 = [r["scen_F1"] for r in rows]
    dl = [r["delay_mean_s"] for r in rows if not np.isnan(r["delay_mean_s"])]
    nd = [r["n_detected"] for r in rows]
    au = [r["auprc"] for r in rows]
    print(f"\n{len(rows)}회 — AUPRC {np.mean(au):.4f} ± {np.std(au):.4f} | "
          f"시나리오 F1 {np.mean(f1):.3f} ± {np.std(f1):.3f} | "
          f"지연 {np.mean(dl):.2f}s ± {np.std(dl):.2f} "
          f"(탐지 {np.mean(nd):.1f}/{rows[0]['n_scenarios']} 평균)")
    print(f"CSV: {outdir}")


if __name__ == "__main__":
    main()
