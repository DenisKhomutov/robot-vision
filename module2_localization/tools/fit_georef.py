import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

LAT_M = 110540.0   # метров в градусе широты
LON_M = 111320.0   # метров в градусе долготы на экваторе (домножается на cos(lat0))


def main():
    ap = argparse.ArgumentParser(description="Геопривязка карты: аффин map(X,Z)->метры (ENU)")
    ap.add_argument("--map", required=True)
    args = ap.parse_args()

    work = ROOT / "maps" / args.map
    ctrl = json.loads((work / "geo_control.json").read_text())
    m = np.load(work / "runtime.npz")
    pos = m["pos"]
    names = [str(x) for x in m["names"]]

    def idx_of(fr):
        for i, n in enumerate(names):
            if f"{fr:05d}" in n:
                return i
        return len(pos) - 1 if fr >= int("".join(filter(str.isdigit, names[-1])) or 0) else None

    S, G = [], []
    for c in ctrl:
        i = idx_of(c["frame"])
        if i is None:
            print(f"кадр {c['frame']} не найден в карте")
            return 1
        S.append([pos[i, 0], pos[i, 2]])
        G.append([c["lat"], c["lon"]])
    S = np.array(S)
    G = np.array(G)

    lat0, lon0 = G.mean(0)
    E = (G[:, 1] - lon0) * np.cos(np.radians(lat0)) * LON_M
    N = (G[:, 0] - lat0) * LAT_M
    T = np.c_[E, N]

    A = np.c_[S, np.ones(len(S))]                      # [X Z 1]
    M, *_ = np.linalg.lstsq(A, T, rcond=None)          # 3x2: map(X,Z)->(E,N)
    res = np.linalg.norm(A @ M - T, axis=1)
    rms = float(np.sqrt((res ** 2).mean()))

    geo = {
        "method": "affine",
        "lat0": float(lat0), "lon0": float(lon0),
        "lat_m": LAT_M, "lon_m": LON_M,
        "affine": M.T.tolist(),                        # 2x3: [[a,b,c],[d,e,f]] строки E,N
        "rms_m": rms,
        "control_frames": [c["frame"] for c in ctrl],
    }
    (work / "geo.json").write_text(json.dumps(geo, ensure_ascii=False, indent=2))
    print(f"опорных точек {len(S)}, СКО {rms:.2f} м")
    print("невязки (м):", [round(float(r), 2) for r in res])
    print(f"-> {work / 'geo.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())