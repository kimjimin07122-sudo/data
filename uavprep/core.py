"""공통 처리 — PRINCIPLES.md P4~P19.

비행 레코드 (reader가 만든다):
  key       비행 이름 (원본 경로 기준)
  group     나누기 묶음 (날짜·세션). 없으면 key
  t         격자 시각 (초, 비행 시작 기준), 길이 T
  x         (T, C) 입력 채널 — 앞 n_state개가 상태(예측 대상), 나머지가 지령
  fault     (T,) bool — 시험 대상 결함 구간 (P12)
  ground    (T,) bool — 지상 (P9)
  info      dict — 재현용 부가 정보 (원본 구간, 결함 시작 초 등)
"""
import json
import os

import numpy as np


# ---------------------------------------------------------------- 격자 (P5, P6)
def grid_times(t0, t1, dt):
    """[t0, t1] 안의 격자. 첫 칸은 t0 + dt (구간 (g-dt, g]가 데이터 안에 있게)."""
    n = int(np.floor((t1 - t0) / dt + 1e-9))
    return t0 + dt * np.arange(1, n + 1)


def resample(t, v, grid, dt, method="mean", with_std=False):
    """원래 주기가 격자보다 빠르면 '직전 dt초 구간 평균'(인과적), 아니면 선형보간 (P6).
    method="last": 모든 채널을 격자 시각 직전(같은 시각 제외)의 마지막 값으로 (Park MATLAB 방식, 비교용).
    빈 구간은 선형보간으로 메운다. 반환: (값, 원래 주기 Hz)
    with_std=True: 직전 dt초 구간 안 표준편차(흔들림 폭)를 세 번째로 돌려준다. 원래 주기가 격자보다 느리면 0."""
    t = np.asarray(t, float)
    v = np.asarray(v, float)
    d = np.diff(t)
    d = d[d > 0]                       # 같은 시각이 두 번 찍힌 행(ArduPilot 로그에 있음)은 간격에서 뺀다
    native_dt = float(np.median(d)) if len(d) > 1 else dt
    if with_std:
        lo = np.searchsorted(t, grid - dt, side="right")
        hi = np.searchsorted(t, grid, side="right")
        cs, cs2 = np.concatenate([[0.0], np.cumsum(v)]), np.concatenate([[0.0], np.cumsum(v * v)])
        cnt = np.maximum(hi - lo, 1)
        m = (cs[hi] - cs[lo]) / cnt
        sd = np.sqrt(np.maximum((cs2[hi] - cs2[lo]) / cnt - m * m, 0.0))
        sd[(hi - lo) < 2] = 0.0
        return resample(t, v, grid, dt, method)[0], 1.0 / native_dt, sd
    if method == "last":
        i = np.searchsorted(t, grid, side="left") - 1
        return v[np.clip(i, 0, None)], 1.0 / native_dt
    if native_dt < dt * 0.9:
        lo = np.searchsorted(t, grid - dt, side="right")      # (g-dt, g]
        hi = np.searchsorted(t, grid, side="right")
        cs = np.concatenate([[0.0], np.cumsum(v)])
        cnt = hi - lo
        with np.errstate(invalid="ignore", divide="ignore"):
            out = (cs[hi] - cs[lo]) / cnt
        empty = cnt == 0
        if empty.any():
            out[empty] = np.interp(grid[empty], t, v)
        return out, 1.0 / native_dt
    return np.interp(grid, t, v), 1.0 / native_dt


def euler_to_quat(roll, pitch, yaw):
    """도(deg) 오일러 → 쿼터니언 (x, y, z, w). scipy Rotation.from_euler('xyz', degrees=True).as_quat()와 같다
    (Park 노트북 data_preprocessing과 같은 규약)."""
    r, p, y = (np.radians(np.asarray(a, float)) / 2 for a in (roll, pitch, yaw))
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    yq = cy * sp * cr + sy * cp * sr
    z = sy * cp * cr - cy * sp * sr
    return np.stack([x, yq, z, w], axis=-1)


def continuous_quat(q):
    """P3 (v0.3): 쿼터니언 부호를 시간축으로 이어 붙인다. q와 −q는 같은 자세라, 직전 값과 내적이 음수면 뒤집는다.
    요를 (−180, 180]으로 접어 변환하면 기체가 180°를 지날 때마다 값이 +1 ↔ −1로 튀어 가짜 예측 오차가 생긴다
    (ALFA 정상 학습·검증 24개 시퀀스에서 64회). v0.4 기본값은 끔 — 켜면 ALFA AUROC가 0.929 → 0.890으로 떨어졌다."""
    q = np.array(q, float, copy=True)
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    return q


# ---------------------------------------------------------------- 구간 나누기 (P9, P10)
def segments(rec, min_len):
    """지상이 아닌 연속 구간을 별도 시퀀스로. 짧은 조각은 버리고 이유를 남긴다."""
    air = ~rec["ground"]
    out, dropped = [], []
    edges = np.flatnonzero(np.diff(np.concatenate([[0], air.astype(int), [0]])))
    for n, (a, b) in enumerate(zip(edges[::2], edges[1::2])):
        seg = dict(rec, key=rec["key"] if len(edges) == 2 else f'{rec["key"]}#{n}',
                   t=rec["t"][a:b], x=rec["x"][a:b], fault=rec["fault"][a:b], ground=rec["ground"][a:b])
        (out if b - a >= min_len else dropped).append(seg)
    return out, [dict(key=s["key"], steps=len(s["t"]), reason=f"지상으로 끊긴 짧은 조각 (< {min_len} 스텝)")
                 for s in dropped]


# ---------------------------------------------------------------- 창 만들기 (P13)
def windows(x, fault, L, n_state):
    n = len(x) - L
    X = np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(x, L, axis=0)[:n].transpose(0, 2, 1))
    return X, x[L:L + n, :n_state].copy(), fault[L:L + n]


# ---------------------------------------------------------------- 저장 (P18)
def build_npz(sets, L, n_state, channels, out_npz, meta):
    """sets: {이름: [레코드…]} — 'normal'이 학습 셋. 학습 셋으로 z-score 후 npz·norm·meta 저장."""
    tr = np.concatenate([r["x"] for r in sets["normal"]])
    mean, std = tr.mean(0), tr.std(0)
    std = np.where(std < 1e-12, 1.0, std)
    out, summary = {}, {}
    for s, recs in sets.items():
        Xs, Ys, Ls, SL, K = [], [], [], [], []
        for r in recs:
            x = ((r["x"] - mean) / std).astype(np.float32)
            X, Y, f = windows(x, r["fault"], L, n_state)
            Xs.append(X); Ys.append(Y); Ls.append(f); SL.append(len(x)); K.append(r["key"])
        lab = np.concatenate(Ls)
        out[f"{s}_X"] = np.concatenate(Xs)
        out[f"{s}_Y"] = np.concatenate(Ys)
        out[f"{s}_label"] = np.stack([~lab, lab], 1).astype(np.float64)
        out[f"{s}_seq_length"] = np.array(SL, np.int64)
        out[f"{s}_keys"] = np.array(K)
        summary[s] = dict(sequences=len(K), windows=int(len(lab)), fault_windows=int(lab.sum()),
                          X_shape=list(out[f"{s}_X"].shape), Y_shape=list(out[f"{s}_Y"].shape))
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez(out_npz, **out)
    norm = out_npz.replace(".npz", "_norm.npz")
    np.savez(norm, mean=mean, std=std, channels=np.array(channels))
    meta = dict(meta, channels=channels, n_state=n_state, L=L, sets=summary,
                files=dict(npz=out_npz, norm=norm))
    with open(out_npz.replace(".npz", "_meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1, default=str)
    return summary


def split_groups(keys_groups, valid_frac, seed):
    """묶음(group) 단위로 섞어 앞에서부터 검증에 넣는다 — 검증 비행 수가 valid_frac에 가장 가까워질 때까지 (P15, P16)."""
    groups = {}
    for k, g in keys_groups:
        groups.setdefault(g, []).append(k)
    names = sorted(groups)
    np.random.default_rng(seed).shuffle(names)
    total = sum(len(v) for v in groups.values())
    valid, n = [], 0
    for g in names:
        if n >= valid_frac * total:
            break
        valid.append(g); n += len(groups[g])
    vset = {k for g in valid for k in groups[g]}
    return sorted(k for k, _ in keys_groups if k not in vset), sorted(vset)
