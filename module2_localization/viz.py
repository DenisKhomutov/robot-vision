"""
    uv run --no-sync python -m module2_localization.viz --map map_rec3 \
        --nats-url nats://<IP-джетсона>:4222
"""
import argparse
import asyncio
import json

import cv2
import numpy as np

from . import config
from .nats_client import NatsClient

SIZE = 900
FONT = cv2.FONT_HERSHEY_SIMPLEX


def build(map_name):
    """Канва (облако + эталонная линия) и функция проекции + позиции узлов маршрута."""
    m = np.load(config.MAPS_DIR / map_name / "runtime.npz")
    pos = dict(zip(m["names"].tolist(), m["pos"]))
    order = sorted(pos)
    rcam = getattr(config, "ROUTE_CAM", None)
    if rcam:
        order = [n for n in order if rcam in n]
    P = np.array([pos[n] for n in order])
    if len(P) > 5:
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        thr = 10 * np.median(seg)
        keep = [True] * len(P)
        for k in range(1, len(P) - 1):
            if seg[k - 1] > thr and seg[k] > thr:
                keep[k] = False
        P = P[keep]
    rn = getattr(config, "ROUTE_NODES", None)
    if rn:
        P = P[:rn]
    xyz = m["points"]
    lo, hi = np.percentile(xyz[:, [0, 2]], [2, 98], axis=0)
    pad = int(SIZE * 0.08)

    def px(p):
        q = (np.atleast_2d(p) - lo) / (hi - lo + 1e-9)
        return np.stack([pad + q[:, 0] * (SIZE - 2 * pad),
                         SIZE - pad - q[:, 1] * (SIZE - 2 * pad)], 1).astype(int)

    canvas = np.full((SIZE, SIZE, 3), 16, np.uint8)
    pp = px(xyz[:, [0, 2]])
    m = (pp[:, 0] >= 0) & (pp[:, 0] < SIZE) & (pp[:, 1] >= 0) & (pp[:, 1] < SIZE)
    for x, y in pp[m]:
        cv2.circle(canvas, (x, y), 1, (70, 70, 70), -1)
    rp = px(P[:, [0, 2]])
    med = np.median(np.linalg.norm(np.diff(P, axis=0), axis=1))
    for k in range(len(rp) - 1):
        if np.linalg.norm(P[k + 1] - P[k]) < 30 * med:
            cv2.line(canvas, tuple(rp[k]), tuple(rp[k + 1]), (200, 170, 60), 2, cv2.LINE_AA)
    cv2.circle(canvas, tuple(rp[0]), 8, (120, 230, 120), -1)
    cv2.circle(canvas, tuple(rp[-1]), 8, (60, 60, 240), -1)
    return canvas, rp, px


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=config.DEFAULT_MAP)
    ap.add_argument("--nats-url", default=f"nats://{config.NATS_HOST}:4222")
    ap.add_argument("--topic", default=config.NATS_TOPIC)
    args = ap.parse_args()

    canvas, rp, px = build(args.map)
    n_nodes = len(rp)
    latest = {"cmd": None}

    async def on_msg(msg):
        try:
            latest["cmd"] = json.loads(msg.data.decode())
        except json.JSONDecodeError:
            pass

    nc = NatsClient(args.nats_url)
    await nc.connect()
    await nc.subscribe(args.topic, on_msg)
    print(f"[viz] карта {args.map}, слушаю {args.topic} на {args.nats_url}", flush=True)

    win = "localization"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    while True:
        img = canvas.copy()
        c = latest["cmd"]
        if c and c.get("move_type") != "lost" and c.get("node") is not None:
            node = min(max(c["node"], 0), n_nodes - 1)
            j = min(node + config.LOOKAHEAD_NODES, n_nodes - 1)
            col = {"left": (60, 200, 255), "right": (60, 200, 255),
                   "stop": (60, 60, 240)}.get(c["move_type"], (80, 255, 80))
            # реальная позиция со сносом вбок; узел — запасной вариант, если pos нет
            if c.get("pos") is not None:
                p = tuple(px(c["pos"])[0])
                cv2.line(img, p, tuple(rp[node]), (90, 90, 90), 1, cv2.LINE_AA)  # снос до линии
                h = c.get("head")
                if h is not None:
                    hp = px([c["pos"][0] + h[0] * 0.5, c["pos"][1] + h[1] * 0.5])[0]
                    cv2.arrowedLine(img, p, tuple(hp), (0, 235, 235), 2, cv2.LINE_AA, tipLength=0.3)
            else:
                p = tuple(rp[node])
                cv2.arrowedLine(img, p, tuple(rp[j]), (0, 235, 235), 2, cv2.LINE_AA, tipLength=0.3)
            cv2.circle(img, p, 9, (60, 60, 255), -1)
            cv2.circle(img, p, 9, (255, 255, 255), 2)
            dm = c.get("dist_to_route_m")
            dtxt = f"{dm:.2f} m" if dm is not None else f"{c.get('dist_to_route', 0):.2f}"
            cv2.putText(img, f"node {node}/{n_nodes}  to route {dtxt}  "
                             f"deg {c.get('deg', 0.0):+.1f}  inl {c.get('inliers', 0)}",
                        (16, 30), FONT, 0.55, (235, 235, 235), 1)
            cv2.putText(img, c["move_type"].upper(), (16, 62), FONT, 0.9, col, 2)
        else:
            reason = (c or {}).get("reason", "нет данных")
            cv2.putText(img, f"LOST ({reason})", (16, 30), FONT, 0.7, (90, 90, 240), 2)
        cv2.imshow(win, img)
        if cv2.waitKey(30) == 27:   # Esc
            break
        await asyncio.sleep(0.001)

    await nc.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main())
