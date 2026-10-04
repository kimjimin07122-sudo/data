"""교수님 데이터셋(Ahn et al., Measurement 2026) reader — 메시지별 CSV + labels.json → 비행 레코드.

※ 서약서 대상 데이터. 원본·라벨(labels.json)·이 reader의 출력(npz 등)은 허가된 PC 밖으로 보내지 않는다 (저장소에도 올리지 않는다).

원칙 적용 (PRINCIPLES.md):
  P2  입력 16 = 상태 10(GPS 속도 3, 자세 q 4, 각속도 3) + 지령 6(지령 자세 q 4, 스로틀 입력·출력)
  P4  구간 = 모델 채널 메시지(ATT, IMU, GPS, CTUN)만의 겹침 (예전 make_csv는 VIBE·TERR 등 진단용까지 겹침)
  P5  격자 5 Hz (가장 느린 핵심 채널 = GPS 5 Hz). L = 32 (6.4 s)
  P6  빠른 채널(ATT·CTUN 10 Hz, IMU 25 Hz)은 직전 0.2 s 평균, GPS는 선형보간
  P3  오일러(도) → 쿼터니언 (x, y, z, w). 요는 펼쳐서 평균한 뒤 (-180, 180]으로 접어 변환 (ALFA와 같음)
  P9  지상 = v2 규칙 (모터 PWM 평균 ≤ 1150, 또는 스로틀 출력 < 0.05 이고 PWM 평균 < 1300) — 9/27 수정 D
  P12 결함 = labels.json ranges (make_labels.py가 Anomaly 폴더 구간·txt·표에서 만든 것)
  P15 정상 비행 5편 → 스텝 수 기준 25%가 검증이 되게 (길이 편차가 커서 편수 기준이면 검증이 학습보다 커짐)
  센서 번호(I)가 섞인 메시지(IMU, GPS)는 가장 많이 기록된 번호만 쓴다
  EMI·Heating은 비행 없이 찍은 대리 기록(논문 Table 7)이라 뺀다
정상 비행은 Normal 폴더, 결함 비행은 Origin 폴더(전체 기록)를 쓴다 (기존 make_csv와 같음).
"""
import json
import os
import re

import numpy as np
import pandas as pd

from ..core import continuous_quat, euler_to_quat, grid_times, resample

STATE = ["vel_x", "vel_y", "vel_z", "q_x", "q_y", "q_z", "q_w", "ang_vel_x", "ang_vel_y", "ang_vel_z"]
COMMAND = ["q_des_x", "q_des_y", "q_des_z", "q_des_w", "throttle_in", "throttle_out"]
CATEGORIES = {"Normal Flight Data": "Normal", "GPS Error Data": "GPS", "IMU Error Data": "IMU",
              "Motor Error Data": "Motor", "Prop Vibration Data": "Vibration", "EMI Error Data": "EMI",
              "Heating Data": "Heating"}


def list_flights(root):
    out = []
    csv_root = os.path.join(root, "CSV Labeling File")
    for cat in sorted(os.listdir(csv_root)):
        cls = CATEGORIES.get(re.sub(r"^\d+\.", "", cat).strip())
        cp = os.path.join(csv_root, cat)
        if cls is None or not os.path.isdir(cp):
            continue
        for fl in sorted(os.listdir(cp)):
            fd = os.path.join(cp, fl)
            if not os.path.isdir(fd):
                continue
            parts = {}
            for e in sorted(os.listdir(fd)):
                p = os.path.join(fd, e)
                if os.path.isdir(p):
                    low = e.lower()
                    parts["origin" if "origin" in low else "normal" if "normal" in low else "anomaly"] = p
            out.append(dict(key=fl, cls=cls, parts=parts))
    return out


def _msg(part, msg, cols):
    for f in sorted(os.listdir(part)):
        stem, ext = os.path.splitext(f)
        if ext.lower() == ".csv" and stem.upper().endswith(msg):
            d = pd.read_csv(os.path.join(part, f))
            d["timestamp"] = pd.to_numeric(d["timestamp"], errors="coerce")
            d = d.dropna(subset=["timestamp"]).sort_values("timestamp")
            if "I" in d.columns:
                d = d[d["I"] == d["I"].mode().iloc[0]]
            return d["timestamp"].to_numpy(float), d[cols].apply(pd.to_numeric, errors="coerce")
    return None


def _cq(cfg, q):
    """P3: quat_continuous(기본 false = v0.4)면 접은 요로 변환한 값 그대로(Park 방식), true면 부호 연속화(v0.3, ALFA·교수님 모두 같거나 나빠 되돌림)."""
    return continuous_quat(q) if cfg.get("quat_continuous", False) else q


def _unwrap_deg(a):
    a = np.asarray(a, float)
    return np.degrees(np.unwrap(np.radians(a)))


def _wrap_deg(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def read(cfg, f, labels):
    part = f["parts"].get("normal" if f["cls"] == "Normal" else "origin") or f["parts"].get("normal")
    att = _msg(part, "ATT", ["Roll", "Pitch", "Yaw", "DesRoll", "DesPitch", "DesYaw"])
    imu = _msg(part, "IMU", ["GyrX", "GyrY", "GyrZ"])
    gps = _msg(part, "GPS", ["Spd", "GCrs", "VZ"])
    ctun = _msg(part, "CTUN", ["ThI", "ThO", "BAlt"])
    rcou = _msg(part, "RCOU", ["C1", "C2", "C3", "C4"])
    if any(m is None for m in (att, imu, gps, ctun)):
        return dict(key=f["key"], error="모델 채널 메시지 없음")
    core = [att, imu, gps, ctun]
    t_first = max(m[0][0] for m in core)
    t_last = min(m[0][-1] for m in core)
    dt = 1.0 / cfg["hz"]
    g = grid_times(t_first, t_last, dt)

    def rs(m, col, unwrap=False):
        t, d = m
        v = d[col].to_numpy(float)
        ok = np.isfinite(v)
        return resample(t[ok], _unwrap_deg(v[ok]) if unwrap else v[ok], g, dt)[0]

    # GPS 속도: 지상속도·방위 → 수평 속도 벡터를 원래 주기에서 만든 뒤 격자로 (방위 접힘을 피함)
    tg, dg = gps
    th = np.radians(dg["GCrs"].to_numpy(float))
    vx, vy = dg["Spd"].to_numpy(float) * np.cos(th), dg["Spd"].to_numpy(float) * np.sin(th)
    vel = [resample(tg, v, g, dt)[0] for v in (vx, vy, dg["VZ"].to_numpy(float))]
    r, p, y = rs(att, "Roll"), rs(att, "Pitch"), _wrap_deg(rs(att, "Yaw", True))
    dr, dp, dy = rs(att, "DesRoll"), rs(att, "DesPitch"), _wrap_deg(rs(att, "DesYaw", True))
    x = np.column_stack(vel + [_cq(cfg, euler_to_quat(r, p, y))] + [rs(imu, c) for c in ("GyrX", "GyrY", "GyrZ")]
                        + [_cq(cfg, euler_to_quat(dr, dp, dy)), rs(ctun, "ThI"), rs(ctun, "ThO")])
    # 지상 (P9, v2 규칙). RCOU가 없는 시각은 지상 아님으로 둔다
    ground = np.zeros(len(g), bool)
    if rcou is not None:
        tr, dr_ = rcou
        pwm = dr_.mean(axis=1).to_numpy(float)
        ok = np.isfinite(pwm)
        pwm_g = resample(tr[ok], pwm[ok], g, dt)[0]
        inside = (g >= tr[0]) & (g <= tr[-1])
        gr = cfg["ground"]
        ground = inside & ((pwm_g <= gr["pwm_max"]) | ((x[:, -1] < gr["idle_throttle_max"]) & (pwm_g < gr["idle_pwm_max"])))
    # 이륙 전(바닥에서 떠오르기 전)도 지상 (P9 v0.3): 모터는 돌아도 아직 땅 위인 과도 구간이 시퀀스 앞에 붙어
    # 비행별 기준선(처음 10 s)을 오염시켰다. 기존 make_csv 규칙: 기압 고도가 지면(처음 30스텝 중앙값)보다
    # liftoff_alt_m 이상 오르고 모터 PWM > pwm_max인 첫 시각부터. 정상 비행은 마지막으로 그 조건이 성립한 시각까지.
    if cfg.get("liftoff_alt_m") and rcou is not None:
        alt = rs(ctun, "BAlt")
        base = np.nanmedian(alt[:30])
        up = ((alt - base) > cfg["liftoff_alt_m"]) & (pwm_g > cfg["ground"]["pwm_max"]) & inside
        if up.any():
            a0 = int(np.argmax(up))
            ground[:a0] = True
            if f["cls"] == "Normal":
                ground[len(up) - int(np.argmax(up[::-1])):] = True
    fault = np.zeros(len(g), bool)
    for a, b in labels.get(f["key"], {}).get("ranges", []):
        fault |= (g >= a) & (g <= b)
    return dict(key=f["key"], group=f["key"], t=g - t_first, x=x, fault=fault, ground=ground,
                info=dict(cls=f["cls"], core_end_s=round(t_last - t_first, 2), air_steps=int((~ground).sum()),
                          onset_s=round(float(g[fault][0] - t_first), 2) if fault.any() else None))


def balanced_valid(recs, frac):
    """긴 비행부터 보며, 검증 스텝이 전체의 frac을 넘지 않는 선에서 검증에 넣는다 (기존 export_npz 규칙)."""
    order = sorted(recs, key=lambda r: -r["info"]["air_steps"])
    total = sum(r["info"]["air_steps"] for r in order)
    valid, got = [], 0
    for r in order:
        if got + r["info"]["air_steps"] <= total * frac:
            valid.append(r); got += r["info"]["air_steps"]
    if not valid:
        valid = [order[-1]]
    return [r for r in order if r not in valid], valid


def load_all(cfg):
    labels = json.load(open(cfg["labels"]))
    sets, excluded = {}, []
    by_cls = {}
    for f in list_flights(cfg["raw"]):
        if f["cls"] in cfg["exclude_classes"]:
            excluded.append(dict(key=f["key"], set=f["cls"].lower(), reason=cfg["exclude_classes"][f["cls"]]))
            continue
        r = read(cfg, f, labels)
        if "error" in r:
            excluded.append(dict(key=f["key"], set=f["cls"].lower(), reason=r["error"]))
            continue
        by_cls.setdefault(f["cls"], []).append(r)
    tr, va = balanced_valid(by_cls.pop("Normal"), cfg["valid_frac"])
    sets["normal"], sets["valid"] = tr, va
    for cls in ("GPS", "IMU", "Motor", "Vibration"):
        sets[cls.lower()] = by_cls.get(cls, [])
    return sets, excluded, STATE, COMMAND
