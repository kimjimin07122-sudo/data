"""수정 B — 저장된 잔차(residuals.csv)로 평가를 여러 조건에서 다시 계산한다. 재학습 없음.

train_eval.py의 평가 함수(evaluate, ranking_metrics)를 그대로 불러와 쓴다. 계산 규칙이 원본과
같아야 "조건만 바꾼 비교"가 되기 때문이다. 먼저 조건 V0(원래 그대로)로 run의 metrics.csv를
재현하는지 확인하고, 다르면 멈춘다.

조건 (사용자 결정 2026-09-27: 결함 후 포함 결과와 제외 결과를 둘 다 보고):
  V0 원래          모든 비행, 모든 스텝 (= 기존 결과, 검증용)
  V1 결함후제외    결함 비행에서 마지막 결함 스텝 이후를 점수 계산에서 뺀다
  V2 목록제외      --exclude 의 비행을 시험셋에서 뺀다 (모델이 볼 수 없는 결함 등).
                   결함 라벨이 하나도 없는 시험 시퀀스도 자동으로 뺀다 — train_eval.evaluate는 이런
                   시퀀스를 '놓침(FN)'으로 세는데, 탐지할 결함이 없으니 원리적으로 맞힐 수 없다.
                   (지상 구간으로 나뉜 조각, 결함이 앞 L스텝 안에만 있는 비행 등)
  V3 둘다          V1 + V2
임계값 T는 검증셋에서 정한 값 그대로다 (시험 조건을 바꿔도 T는 안 바뀐다).
탐지 지연은 --dt(데이터의 실제 샘플 간격)로 다시 계산한다 — run이 잘못된 dt로 돌았어도 바로잡힌다.

사용 (torch가 있는 환경):
  python models/eval_variants.py --run <run 폴더> --dt 0.2 \
      --exclude <exclude.json> --out <결과 폴더>
--labels-npz: 잔차는 그대로 두고 결함 라벨만 다른 npz의 것으로 바꿔 평가한다(라벨 정의만 바꾼 효과를 재학습 없이 분리).
exclude.json의 "exclude_eval" 키는 "<set>_<번호3자리>"(npz 순번) 또는 npz keys의 비행 이름.
비행 이름으로 적으면 --npz 로 keys를 읽어 순번으로 바꾼다. 세그먼트 키('<비행>#<n>')는 비행 이름으로 묶인다.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_eval as te  # noqa: E402  (evaluate, ranking_metrics, DT, FAULT_SETS를 쓴다)

VARIANTS = ["V0_원래", "V1_결함후제외", "V2_목록제외", "V3_둘다"]


def load_seed(path, L):
    d = pd.read_csv(path, usecols=["set", "scenario", "step", "residual", "threshold", "fault"])
    d = d.sort_values(["set", "scenario", "step"])
    T = float(d.threshold.iloc[0])
    per = {}
    for (s, k), g in d.groupby(["set", "scenario"], sort=True):
        per.setdefault(s, []).append((g.residual.to_numpy(), g.fault.to_numpy().astype(bool)))
    return per, T


def build(per, sets, L, drop_post, drop_idx, drop_nofault=False):
    """조건에 맞게 잘라서 train_eval이 받는 꼴(residuals dict, data dict)로 만든다."""
    residuals, data = {}, {}
    for s in sets:
        rs, fs, sl = [], [], []
        for k, (r, f) in enumerate(per[s]):
            if (s, k) in drop_idx or (drop_nofault and not f.any()):
                continue
            if drop_post and f.any():
                end = len(f) - int(np.argmax(f[::-1]))      # 마지막 결함 스텝 다음
                r, f = r[:end], f[:end]
            rs.append(r); fs.append(f); sl.append(len(r) + L)
        residuals[s] = np.concatenate(rs) if rs else np.zeros(0)
        data[s] = dict(fault=np.concatenate(fs) if fs else np.zeros(0, bool),
                       seq_length=np.array(sl, dtype=np.int64))
    return residuals, data


def resolve_exclude(path, npz):
    if not path:
        return {}, {}
    ex = json.load(open(path)).get("exclude_eval", {})
    idx, why = set(), {}
    keys = {}
    if npz:
        z = np.load(npz, allow_pickle=True)
        keys = {f[:-5]: [str(x) for x in z[f]] for f in z.files if f.endswith("_keys")}
    for name, reason in ex.items():
        hit = False
        if "_" in name and name.rsplit("_", 1)[1].isdigit() and name.rsplit("_", 1)[0] in (
                "mechanical", "gps", "imu", "motor", "vibration", "engine", "elevator", "rudder"):
            s, k = name.rsplit("_", 1)
            idx.add((s, int(k))); why[(s, int(k))] = reason; hit = True
        for s, ks in keys.items():
            for k, key in enumerate(ks):
                if key == name or key.split("#")[0] == name:
                    idx.add((s, k)); why[(s, k)] = reason; hit = True
        if not hit:
            if keys:     # keys가 있는데 없다 = 전처리(짧은 조각 버림 등)로 이미 데이터에서 빠졌다
                print(f"  제외 목록 '{name}': 이 npz에 이미 없음 (전처리에서 빠짐) — 건너뜀")
                continue
            raise SystemExit(f"exclude 항목 '{name}'을 찾지 못했다 (--npz 로 keys를 주었는지 확인)")
    return idx, why


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--dt", type=float, required=True, help="데이터의 실제 샘플 간격(초)")
    ap.add_argument("--exclude", default="", help="exclude.json 경로")
    ap.add_argument("--npz", default="", help="exclude를 비행 이름으로 적었을 때 keys를 읽을 npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--labels-npz", default="",
                    help="결함 라벨만 이 npz의 <결함셋>_label로 바꿔 평가한다(잔차·임계값은 run 그대로). "
                         "재현 확인은 원래 라벨로 먼저 하고, 표는 바꾼 라벨로 계산한다. 시퀀스 길이가 같아야 한다.")
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.run, "config.json")))
    L = cfg["L"]
    fault_sets = (cfg.get("fault_sets") or "engine,elevator,rudder").split(",")
    te.FAULT_SETS = fault_sets
    te.DT = args.dt
    orig = pd.read_csv(os.path.join(args.run, "metrics.csv"))
    drop_idx, why = resolve_exclude(args.exclude, args.npz)

    print(f"run {args.run}\n  L={L}, 결함 셋 {fault_sets}, run dt {cfg.get('dt')} → 평가 dt {args.dt}")
    for (s, k), w in sorted(why.items()):
        print(f"  제외 목록: {s}_{k:03d} — {w}")

    per0, _ = load_seed(os.path.join(args.run, f"seed{int(orig.seed.iloc[0])}", "residuals.csv"), L)
    nofault = [(s, k) for s in fault_sets for k, (_, f) in enumerate(per0[s])
               if not f.any() and (s, k) not in drop_idx]
    n_all = sum(len(per0[s]) for s in fault_sets)
    print(f"  시험 시퀀스 {n_all}개 중 목록 제외 {len(drop_idx)}개, 그 밖에 결함 라벨 없는 시퀀스 {len(nofault)}개"
          + (": " + ", ".join(f"{s}_{k:03d}" for s, k in nofault) if nofault else ""))

    newlab = {}
    if args.labels_npz:
        z = np.load(args.labels_npz, allow_pickle=True)
        for s in fault_sets:
            lab = z[f"{s}_label"][:, 1].astype(bool)
            sl = z[f"{s}_seq_length"] - L
            if len(sl) != len(per0[s]) or any(len(per0[s][k][1]) != n for k, n in enumerate(sl)):
                raise SystemExit(f"--labels-npz {s}: 시퀀스 수·길이가 run과 다르다")
            newlab[s] = np.split(lab, np.cumsum(sl)[:-1])
            print(f"  라벨 교체 ({s}): 결함 스텝 {sum(int(f.sum()) for _, f in per0[s])} → {int(lab.sum())}  ← {args.labels_npz}")

    rows = []
    for _, o in orig.iterrows():
        seed = int(o.seed)
        per, T = load_seed(os.path.join(args.run, f"seed{seed}", "residuals.csv"), L)
        if newlab:                                        # 원래 라벨로 재현 확인 먼저
            r0, d0 = build(per, fault_sets, L, drop_post=False, drop_idx=set())
            m0 = te.evaluate(r0, d0, L, T)
            a0, p0, _ = te.ranking_metrics(r0, d0, fault_sets)
            got = dict(auroc=a0, auprc=p0, scen_F1=m0["scen_F1"], scen_TP=m0["scen_TP"], scen_FP=m0["scen_FP"])
            bad = {k: (got[k], o[k]) for k in got if k in o.index and not np.isclose(got[k], o[k], atol=1e-6)}
            if bad:
                raise SystemExit(f"seed {seed}: 원래 라벨 V0가 metrics.csv와 다르다 {bad}")
            per = {s: ([(r, newlab[s][k]) for k, (r, _) in enumerate(v)] if s in newlab else v) for s, v in per.items()}
        for v in VARIANTS:
            v2 = v in ("V2_목록제외", "V3_둘다")
            residuals, data = build(per, fault_sets, L,
                                    drop_post=v in ("V1_결함후제외", "V3_둘다"),
                                    drop_idx=drop_idx if v2 else set(), drop_nofault=v2)
            m = te.evaluate(residuals, data, L, T)
            auroc, auprc, pos = te.ranking_metrics(residuals, data, fault_sets)
            row = dict(seed=seed, variant=v, threshold=T, auroc=auroc, auprc=auprc, pos_rate=pos,
                       scen_TP=m["scen_TP"], scen_FP=m["scen_FP"], scen_FN=m["scen_FN"],
                       scen_F1=m["scen_F1"], n_scenarios=m["n_scenarios"],
                       delay_mean_s=m["delay_mean"], ts_F1=m["timestep"]["전체"][2])
            for s in fault_sets:                          # 계열별 순위 지표
                a, p, pr = te.ranking_metrics(residuals, data, [s])
                row[f"{s}_auroc"], row[f"{s}_auprc"], row[f"{s}_pos"] = a, p, pr
            rows.append(row)
            if v == "V0_원래" and not newlab:             # 원래 결과 재현 확인 (라벨 교체 시엔 위에서 함)
                # 옛 run(ALFA paper 등)은 metrics.csv에 auprc·auroc가 없다 → 있는 열로만 확인
                chk = {k: o[k] for k in ("auprc", "auroc", "scen_F1", "scen_TP", "scen_FP") if k in o.index}
                checked = sorted(chk)
                bad = {k: (row[k], x) for k, x in chk.items() if not np.isclose(row[k], x, atol=1e-6)}
                if bad:
                    raise SystemExit(f"seed {seed}: V0가 metrics.csv와 다르다 {bad}")

    df = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    df.to_csv(os.path.join(args.out, f"B_metrics{tag}.csv"), index=False)
    cols = ["auprc", "auroc", "scen_F1", "scen_TP", "scen_FP", "scen_FN", "n_scenarios", "delay_mean_s",
            "pos_rate"] + [c for c in df.columns if c.endswith(("_auprc", "_auroc"))]
    summ = df.groupby("variant", sort=False)[cols].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reset_index()
    summ.to_csv(os.path.join(args.out, f"B_summary{tag}.csv"), index=False)

    if newlab:
        print(f"  원래 라벨 V0가 run의 metrics.csv와 {len(orig)}개 seed 모두 일치 ✓ → 표는 바꾼 라벨로 계산")
    else:
        print(f"  V0(원래)가 run의 metrics.csv와 {len(orig)}개 seed 모두 일치 ✓ (확인한 열: {', '.join(checked)})")
    print(f"\n  {'조건':12s} {'AUPRC':>15s} {'AUROC':>7s} {'시나리오F1':>15s} {'탐지':>9s} {'오경보':>6s} {'지연(s)':>7s} {'양성비율':>7s}")
    for _, r in summ.iterrows():
        print(f"  {r.variant:12s} {r.auprc_mean:.4f} ± {r.auprc_std:.4f} {r.auroc_mean:7.4f} "
              f"{r.scen_F1_mean:.3f} ± {r.scen_F1_std:.3f} {r.scen_TP_mean:4.1f}/{r.n_scenarios_mean:<4.0f} "
              f"{r.scen_FP_mean:6.1f} {r.delay_mean_s_mean:7.2f} {r.pos_rate_mean:7.3f}")
    if len(fault_sets) > 1:
        print(f"\n  계열별 AUPRC (seed 평균)")
        print("  " + f"{'조건':12s}" + "".join(f"{s:>12s}" for s in fault_sets))
        for _, r in summ.iterrows():
            print("  " + f"{r.variant:12s}" + "".join(f"{r[f'{s}_auprc_mean']:12.4f}" for s in fault_sets))
    print(f"\n  → {args.out}/B_metrics{tag}.csv, B_summary{tag}.csv")


if __name__ == "__main__":
    main()
