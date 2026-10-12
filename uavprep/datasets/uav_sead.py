"""UAV-SEAD reader — .ulg + mapping.json → 비행 레코드.

원칙 적용 (PRINCIPLES.md):
  P2  입력 18 = 상태 10(속도 3, 자세 q 4, 각속도 3) + 지령 8(지령 각속도 3, 추력 1, 지령 자세 q 4)
  P4  구간 = 18채널 출처 토픽만의 겹침 (예전 make_csv는 여분 채널까지 겹쳐 결함 무렵이 잘렸다)
  P5  격자 10 Hz (가장 느린 핵심 채널 = vehicle_local_position, vehicle_attitude_setpoint 10 Hz)
  P6  빠른 채널은 직전 0.1 s 평균, 같거나 느린 채널은 선형보간
  P7  핵심 채널 원래 주기가 격자의 절반(5 Hz)보다 느린 로그는 뺀다 (1 Hz 기록 11편)
  P9  지상 = PX4 착지 판정(vehicle_land_detected.landed) 또는 시동 꺼짐(vehicle_status.arming_state != 2)
  P12 결함 = mapping.json에서 label_classes(기본 Mechanical) 주석 구간만
  P16 정상 비행은 날짜 묶음(키의 첫 폴더) 단위로 학습/검증 배정
  그 밖: 학습·검증 후보 중 최대 |속도| > max_vel(20 m/s)인 비행은 뺀다 (EKF 발산, 기존 export_npz 규칙)
자세 q는 PX4 순서 (w, x, y, z).
"""
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ..core import continuous_quat, grid_times, resample, split_groups

STATE = ["vel_meas_x", "vel_meas_y", "vel_meas_z", "q_meas_w", "q_meas_x", "q_meas_y", "q_meas_z",
         "ang_vel_meas_x", "ang_vel_meas_y", "ang_vel_meas_z"]
COMMAND = ["rate_com_x", "rate_com_y", "rate_com_z", "thrust_com", "q_com_w", "q_com_x", "q_com_y", "q_com_z"]
# 채널 → [(토픽, 필드, 부호), …] 앞에서부터 있는 것을 쓴다 (PX4 버전에 따라 다름)
SOURCES = {
    "vel_meas_x": [("vehicle_local_position", "vx", 1)], "vel_meas_y": [("vehicle_local_position", "vy", 1)],
    "vel_meas_z": [("vehicle_local_position", "vz", 1)],
    **{f"q_meas_{c}": [("vehicle_attitude", f"q[{i}]", 1)] for i, c in enumerate("wxyz")},
    **{f"ang_vel_meas_{c}": [("vehicle_angular_velocity", f"xyz[{i}]", 1), ("vehicle_attitude", f"{n}speed", 1),
                             ("sensor_combined", f"gyro_rad[{i}]", 1), ("rate_ctrl_status", f"{n}speed", 1)]
       for i, (c, n) in enumerate(zip("xyz", ["roll", "pitch", "yaw"]))},
    "rate_com_x": [("vehicle_rates_setpoint", "roll", 1)], "rate_com_y": [("vehicle_rates_setpoint", "pitch", 1)],
    "rate_com_z": [("vehicle_rates_setpoint", "yaw", 1)],
    # thrust_body[2]는 NED z축이라 부호가 반대 → -1
    "thrust_com": [("vehicle_rates_setpoint", "thrust", 1), ("vehicle_rates_setpoint", "thrust_body[2]", -1),
                   ("vehicle_attitude_setpoint", "thrust", 1), ("vehicle_attitude_setpoint", "thrust_body[2]", -1)],
    **{f"q_com_{c}": [("vehicle_attitude_setpoint", f"q_d[{i}]", 1)] for i, c in enumerate("wxyz")},
}
TOPICS = sorted({s[0] for v in SOURCES.values() for s in v} | {"vehicle_land_detected", "vehicle_status", "actuator_outputs"})
ACT = ["act_mean", "act_spread"]
STD = ["gyr_std_x", "gyr_std_y", "gyr_std_z", "acc_std_x", "acc_std_y", "acc_std_z"]   # std_outputs 옵션: 격자 한 칸 안 흔들림 폭   # actuator_outputs 옵션: 모터 수와 무관한 요약 (평균, 최대−최소)


def _actuator(ulog):
    """actuator_outputs(0번) 중 모터 채널만 골라 평균·최대−최소. 모터 = 값이 800~2200 안에서만 움직이고 변하는 채널
    (기체 12001은 8개, 4014는 4개. 보조 출력은 0 또는 1500 고정이라 빠진다)."""
    try:
        d = next(x for x in ulog.data_list if x.name == "actuator_outputs" and x.multi_id == 0)
    except StopIteration:
        return None
    o = np.column_stack([d.data[f"output[{i}]"] for i in range(8) if f"output[{i}]" in d.data]).astype(float)
    m = (o.min(0) >= 800) & (o.max(0) <= 2200) & (o.std(0) > 5)
    if m.sum() < 3:
        return None
    o = o[:, m]
    return d.data["timestamp"].astype(float) / 1e6, o.mean(1), o.max(1) - o.min(1), int(m.sum())


def _series(ulog, ch):
    for topic, field, sign in SOURCES[ch]:
        try:
            d = ulog.get_dataset(topic)
        except Exception:
            continue
        if field in d.data and len(d.data["timestamp"]) > 2:
            return d.data["timestamp"].astype(float) / 1e6, sign * d.data[field].astype(float), f"{topic}.{field}"
    return None


def _hold(t, v, g, default):
    i = np.searchsorted(t, g, side="right") - 1
    return np.where(i >= 0, v[np.clip(i, 0, None)], default)


def read(args):
    cfg, key, ranges = args
    from pyulog import ULog
    try:
        ulog = ULog(os.path.join(cfg["raw"], key + ".ulg"), TOPICS)
    except Exception as e:
        return dict(key=key, error=f"ULog 읽기 실패: {e}")
    ser = {ch: _series(ulog, ch) for ch in STATE + COMMAND}
    miss = [c for c, s in ser.items() if s is None]
    if miss:
        return dict(key=key, error=f"채널 없음: {', '.join(miss)}")
    t_first = max(s[0][0] for s in ser.values())
    t_last = min(s[0][-1] for s in ser.values())
    dt = 1.0 / cfg["hz"]
    native = {c: 1.0 / float(np.median(np.diff(s[0]))) for c, s in ser.items()}
    slow = {c: round(h, 2) for c, h in native.items() if h < cfg["hz"] / 2}
    if slow:
        return dict(key=key, error=f"저주기 기록 (P7): {min(slow.values())} Hz < {cfg['hz'] / 2} Hz")
    g = grid_times(t_first, t_last, dt)
    if len(g) < 2:
        return dict(key=key, error="겹치는 구간 없음")
    x = np.column_stack([resample(ser[c][0], ser[c][1], g, dt)[0] for c in STATE + COMMAND])
    if cfg.get("quat_continuous", False):                          # P3: v0.4는 PX4 값 그대로
        x[:, 3:7] = continuous_quat(x[:, 3:7])
        x[:, 14:18] = continuous_quat(x[:, 14:18])
    n_motor = None
    if cfg.get("actuator_outputs"):                                # 예측 대상에 모터 출력 요약 2채널 추가
        a = _actuator(ulog)
        if a is None:
            return dict(key=key, error="actuator_outputs 모터 채널 없음")
        n_motor = a[3]
        act = np.column_stack([resample(a[0], a[1], g, dt)[0], resample(a[0], a[2], g, dt)[0]])
        x = np.column_stack([x[:, :len(STATE)], act, x[:, len(STATE):]])
    if cfg.get("std_outputs"):                                     # 예측 대상에 흔들림 폭 6채널 추가
        acc = []
        for i in range(3):
            try:
                d = ulog.get_dataset("sensor_combined")
                acc.append((d.data["timestamp"].astype(float) / 1e6, d.data[f"accelerometer_m_s2[{i}]"].astype(float)))
            except Exception:
                return dict(key=key, error="sensor_combined 가속도 없음")
        sds = [resample(ser[f"ang_vel_meas_{c}"][0], ser[f"ang_vel_meas_{c}"][1], g, dt, with_std=True)[2] for c in "xyz"]
        sds += [resample(t_, v_, g, dt, with_std=True)[2] for t_, v_ in acc]
        n_state = len(STATE) + (len(ACT) if cfg.get("actuator_outputs") else 0)
        x = np.column_stack([x[:, :n_state], np.column_stack(sds), x[:, n_state:]])
    # 지상 (P9)
    landed = armed = None
    try:
        d = ulog.get_dataset("vehicle_land_detected")
        landed = _hold(d.data["timestamp"] / 1e6, d.data["landed"].astype(int), g, 1)
    except Exception:
        pass
    try:
        d = ulog.get_dataset("vehicle_status")
        armed = _hold(d.data["timestamp"] / 1e6, (d.data["arming_state"] == 2).astype(int), g, 0)
    except Exception:
        pass
    ground = np.zeros(len(g), bool)
    if landed is not None:
        ground |= landed == 1
    if armed is not None:
        ground |= armed == 0
    fault = np.zeros(len(g), bool)
    for a, b in ranges:
        fault |= (g >= a / 1e6) & (g <= b / 1e6)
    on = g[fault][0] - t_first if fault.any() else None
    return dict(key=key, group=key.split("/")[0] if "/" in key else f"nodate:{key}", t=g - t_first, x=x,
                fault=fault, ground=ground,
                info=dict(core_end_s=round(t_last - t_first, 2), onset_s=round(on, 2) if on is not None else None,
                          max_abs_vel=round(float(np.abs(x[:, :3]).max()), 2),
                          ground_source=("landed" if landed is not None else "") + ("+armed" if armed is not None else ""),
                          angvel_source=ser["ang_vel_meas_x"][2], thrust_source=ser["thrust_com"][2], n_motor=n_motor))


def load_all(cfg):
    mp = json.load(open(cfg["mapping"]))
    keys = []
    for root, _, files in os.walk(cfg["raw"]):
        for f in files:
            p = os.path.join(root, f)
            if f.endswith(".ulg") and os.path.getsize(p) > 0:
                keys.append(os.path.relpath(p, cfg["raw"])[:-4])
    cls = {k: {a.get("class") for a in mp[k]["annotations"]} for k in keys if k in mp}
    normal = sorted(k for k, c in cls.items() if c == {"Normal"})
    test = sorted(k for k, c in cls.items() if cfg["test_class"] in c)

    def rng(k):
        return [iv for a in mp[k]["annotations"] if a.get("class") in cfg["label_classes"]
                for _, ivs in a.get("ranges", []) for iv in ivs]

    jobs = [(cfg, k, rng(k)) for k in normal + test]
    with ProcessPoolExecutor(cfg.get("workers", 12)) as ex:
        recs = {r["key"]: r for r in ex.map(read, jobs, chunksize=4)}
    excluded = []
    for k, r in recs.items():
        if "error" in r:
            excluded.append(dict(key=k, set="test" if k in test else "normal", reason=r["error"]))
    ok = {k: r for k, r in recs.items() if "error" not in r}
    cand = []
    for k in normal:
        if k not in ok:
            continue
        if ok[k]["info"]["max_abs_vel"] > cfg["max_vel"]:
            excluded.append(dict(key=k, set="normal", reason=f"최대 |속도| {ok[k]['info']['max_abs_vel']} m/s > {cfg['max_vel']} (EKF 발산)"))
            continue
        cand.append((k, ok[k]["group"]))
    if cfg.get("split_from"):          # 비교 실험용: 다른 npz의 학습/검증 배정을 그대로 (빠진 비행만 빠짐)
        z = np.load(cfg["split_from"], allow_pickle=True)
        base = {s_: {str(k).split("#")[0] for k in z[f"{s_}_keys"]} for s_ in ("normal", "valid")}
        ck = {k for k, _ in cand}
        tr, va = sorted(base["normal"] & ck), sorted(base["valid"] & ck)
    else:
        tr, va = split_groups(cand, cfg["valid_frac"], cfg["seed"])
    sets = {"normal": [ok[k] for k in tr], "valid": [ok[k] for k in va],
            "mechanical": [ok[k] for k in test if k in ok]}
    return (sets, excluded, STATE + (ACT if cfg.get("actuator_outputs") else []) + (STD if cfg.get("std_outputs") else []),
            COMMAND)
