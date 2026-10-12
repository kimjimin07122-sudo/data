"""ALFA reader — 공식 processed 배포본(토픽별 CSV) → 비행 레코드.

원칙 적용 (PRINCIPLES.md):
  P4  구간 = 모델 토픽(nav_info-velocity, nav_info-roll/pitch/yaw, local_position-velocity)이 겹치는 시간
  P5  격자 4 Hz (가장 느린 핵심 채널 = 각속도 4.1 Hz)
  P6  20 Hz 토픽은 직전 0.25 s 평균, 각속도는 선형보간. 요는 펼쳐서(unwrap) 평균한 뒤 다시 (-180, 180]으로 접어 변환
  P3  오일러(도) → 쿼터니언 (x, y, z, w) — Park과 같은 규약
  P9  지상 없음 (원본이 자동비행 구간만)
  P12 결함 = 결함 주입 신호(failure_status-*)가 처음 온 시각부터 끝까지
  ALFA 전용: 시퀀스 처음 drop_head_s(8 s)를 버린다 — Park(ICRA 논문 V.B) "제어기가 안정되는 데 8초"
  정상 셋에 쓰는 결함 비행은 결함 시작 전까지만 (Park과 같음)
Park과 다른 점: 마지막 값 → 구간 평균, 피치·요를 자기 시각으로 맞춤(롤 순번 아님), 구간 = 핵심 토픽 겹침.
"""
import os

import numpy as np
import pandas as pd

from ..core import continuous_quat, euler_to_quat, grid_times, resample

STD = ["gyr_std_x", "gyr_std_y", "gyr_std_z", "acc_std_x", "acc_std_y", "acc_std_z"]   # std_outputs: imu-data_raw(10 Hz) 칸 안 흔들림 폭
STATE = ["vel_meas_x", "vel_meas_y", "vel_meas_z", "q_meas_x", "q_meas_y", "q_meas_z", "q_meas_w",
         "angvel_x", "angvel_y", "angvel_z"]
COMMAND = ["vel_com_x", "vel_com_y", "vel_com_z", "q_com_x", "q_com_y", "q_com_z", "q_com_w"]
FAIL = ["engines", "aileron", "rudder", "elevator"]


def _load(raw, seq, topic, cols):
    p = os.path.join(raw, seq, f"{seq}-{topic}.csv")
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p, usecols=["%time"] + cols)
    t = d.pop("%time").to_numpy(np.int64) / 1e9
    return t, d


def _cq(cfg, q):
    """P3: quat_continuous(기본 false = v0.4)면 접은 요로 변환한 값 그대로(Park 방식), true면 부호 연속화(v0.3, ALFA·교수님 모두 같거나 나빠 되돌림)."""
    return continuous_quat(q) if cfg.get("quat_continuous", False) else q


def _unwrap_deg(a):
    return np.degrees(np.unwrap(np.radians(a)))


def _wrap_deg(a):
    """(-180, 180]로 다시 접는다. 펼친 채로 쿼터니언을 만들면 한 바퀴마다 부호가 뒤집혀(q와 -q)
    같은 자세가 다른 값이 된다 → 접은 뒤 변환해 Park과 같은 표현(w ≥ 0에 가까움)을 쓴다."""
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def read(cfg, seq, cut_at_onset):
    raw = cfg["raw"]
    vel = _load(raw, seq, "mavros-nav_info-velocity", ["field.meas_x", "field.meas_y", "field.meas_z",
                                                       "field.des_x", "field.des_y", "field.des_z"])
    att = {a: _load(raw, seq, f"mavros-nav_info-{a}", ["field.measured", "field.commanded"]) for a in ("roll", "pitch", "yaw")}
    ang = _load(raw, seq, "mavros-local_position-velocity", ["field.twist.angular.x", "field.twist.angular.y",
                                                            "field.twist.angular.z"])
    core = [vel, ang] + list(att.values())
    t_first = max(c[0][0] for c in core)
    t_last = min(c[0][-1] for c in core)
    fails = [_load(raw, seq, f"failure_status-{f}", ["field.data"]) for f in FAIL]
    onset = min((f[0][0] for f in fails if f is not None and len(f[0])), default=None)

    dt = 1.0 / cfg["hz"]
    if cfg.get("grid_origin", "core") == "all_topics":      # 비교용: Park MATLAB처럼 모든 토픽 중 가장 이른 시각 기준
        import glob as _g
        t_all = min(pd.read_csv(f, usecols=["%time"], nrows=1)["%time"].iloc[0] / 1e9
                    for f in _g.glob(os.path.join(raw, seq, "*.csv")))
        t_first = t_all + np.ceil((t_first - t_all) / dt - 1e-9) * dt   # 격자 위상을 Park과 맞춘다
    t0 = t_first + cfg["drop_head_s"]
    t1 = t_last
    if cut_at_onset and onset is not None:
        t1 = min(t1, onset - 1e-6)             # 정상 셋: 결함 시작 전까지
    g = grid_times(t0, t1, dt)
    rates = {}

    def rs(src, col, name, unwrap=False):
        t, d = src
        v = d[col].to_numpy(float)
        out, hz = resample(t, _unwrap_deg(v) if unwrap else v, g, dt, cfg.get("downsample", "mean"))
        rates[name] = round(hz, 2)
        return out

    if cfg.get("pitch_yaw_time", "own") == "roll":            # 비교용: Park MATLAB처럼 롤 메시지 순번으로 꺼냄
        n = min(len(att[a][0]) for a in att)
        att = {a: (att["roll"][0][:n], att[a][1].iloc[:n]) for a in att}
    m = {a: rs(att[a], "field.measured", f"{a}_meas", a == "yaw") for a in att}
    c = {a: rs(att[a], "field.commanded", f"{a}_com", a == "yaw") for a in att}
    m["yaw"], c["yaw"] = _wrap_deg(m["yaw"]), _wrap_deg(c["yaw"])   # 평균은 펼쳐서, 변환은 접어서
    x = np.column_stack([
        rs(vel, "field.meas_x", "vel_meas_x"), rs(vel, "field.meas_y", "vel_meas_y"), rs(vel, "field.meas_z", "vel_meas_z"),
        _cq(cfg, euler_to_quat(m["roll"], m["pitch"], m["yaw"])),
        rs(ang, "field.twist.angular.x", "angvel_x"), rs(ang, "field.twist.angular.y", "angvel_y"),
        rs(ang, "field.twist.angular.z", "angvel_z"),
        rs(vel, "field.des_x", "vel_com_x"), rs(vel, "field.des_y", "vel_com_y"), rs(vel, "field.des_z", "vel_com_z"),
        _cq(cfg, euler_to_quat(c["roll"], c["pitch"], c["yaw"])),
    ])
    if cfg.get("std_outputs"):
        imu = _load(raw, seq, "mavros-imu-data_raw", [f"field.angular_velocity.{c}" for c in "xyz"] + [f"field.linear_acceleration.{c}" for c in "xyz"])
        sds = [resample(imu[0], imu[1][c].to_numpy(float), g, dt, with_std=True)[2] for c in imu[1].columns]
        x = np.column_stack([x[:, :len(STATE)], np.column_stack(sds), x[:, len(STATE):]])
    fault = (g >= onset) if onset is not None else np.zeros(len(g), bool)
    return dict(key=seq.replace("carbonZ_", ""), group=seq.replace("carbonZ_", ""), t=g - t_first, x=x,
                fault=fault, ground=np.zeros(len(g), bool),
                info=dict(core_start_s=0.0, core_end_s=round(t_last - t_first, 2),
                          onset_s=round(onset - t_first, 2) if onset is not None else None,
                          native_hz=rates))


def load_all(cfg):
    """반환: (sets {이름: [레코드]}, excluded [사유])"""
    sets, excluded = {}, []
    for name, spec in cfg["sets"].items():
        recs = []
        for seq in spec["sequences"]:
            if seq in cfg.get("exclude", {}):
                excluded.append(dict(key=seq, set=name, reason=cfg["exclude"][seq]))
                continue
            recs.append(read(cfg, "carbonZ_" + seq, cut_at_onset=spec.get("cut_at_onset", False)))
        sets[name] = recs
    return sets, excluded, STATE + (STD if cfg.get("std_outputs") else []), COMMAND
