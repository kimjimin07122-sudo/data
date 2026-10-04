"""설정 파일 하나로 데이터셋 하나를 만든다.  python -m uavprep.build uavprep/configs/<데이터셋>.json"""
import datetime
import importlib
import json
import sys

from .core import build_npz, segments


def main():
    cfg_path = sys.argv[1]
    cfg = json.load(open(cfg_path))
    reader = importlib.import_module(f"uavprep.datasets.{cfg['dataset']}")
    sets, excluded, state, command = reader.load_all(cfg)
    L = cfg["L"]
    min_len = L + cfg.get("min_extra_steps", 10)
    final, short = {}, []
    for s, recs in sets.items():
        out = []
        for r in recs:
            segs, drop = segments(r, min_len)
            out += segs
            short += [dict(d, set=s) for d in drop]
            if not segs and not drop:
                short.append(dict(key=r["key"], set=s, steps=len(r["t"]), reason="전부 지상"))
        final[s] = out
    meta = dict(dataset=cfg["dataset"], config=cfg_path, built=datetime.datetime.now().isoformat(timespec="seconds"),
                hz=cfg["hz"], principles_version=cfg.get("principles_version"),
                excluded=excluded, dropped_segments=short,
                sequences={s: [dict(key=r["key"], steps=len(r["t"]), fault_steps=int(r["fault"].sum()),
                                    ground_steps=int(r["ground"].sum()), **r["info"]) for r in recs]
                           for s, recs in sets.items()})
    summ = build_npz(final, L, len(state), state + command, cfg["out_npz"], meta)
    print(f"{cfg['dataset']}: {cfg['hz']} Hz, L={L}, 입력 {len(state + command)} / 출력 {len(state)}")
    for s, v in summ.items():
        print(f"  {s:12s} 시퀀스 {v['sequences']:4d}  X{tuple(v['X_shape'])}  결함 윈도 {v['fault_windows']}")
    print(f"  제외 {len(excluded)}개, 짧은 조각 버림 {len(short)}개 → {cfg['out_npz']}")


if __name__ == "__main__":
    main()
