import argparse
import asyncio
import json

import cv2
import numpy as np

from . import config

SIZE = 900
FONT = cv2.FONT_HERSHEY_SIMPLEX


def build(map_name):
    m = np.load(config.MAPS_DIR / map_name / "runtime.npz")
    order = sorted(range(len(m["names"])), key=lambda i: m["names"][i])
    rcam = getattr(config, "ROUTE_CAM", None)
    if rcam:
        order = [i for i in order if rcam in str(m["names"][i])]
    P = m["pos"][order]
    F = m["fwd"][order]
    if len(P) > 5:
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        thr = 10 * np.median(seg)
        keep = np.ones(len(P), bool)
        for k in range(1, len(P) - 1):
            if seg[k - 1] > thr and seg[k] > thr:
                keep[k] = False
        P, F = P[keep], F[keep]
    rn = getattr(config, "ROUTE_NODES", None)
    if rn:
        P, F = P[:rn], F[:rn]


    xyz = m["points"]
    lo, hi = np.percentile(xyz[:, [0, 2]], [2, 98], axis=0)
    pad = int(SIZE * 0.08)

    def px(p):
        q = (np.atleast_2d(p) - lo) / (hi - lo + 1e-9)
        return np.stack([pad + q[:, 0] * (SIZE - 2 * pad),
                         SIZE - pad - q[:, 1] * (SIZE - 2 * pad)], 1).astype(int)

    canvas = np.full((SIZE, SIZE, 3), 16, np.uint8)
    pp = px(xyz[:, [0, 2]])
    vis = (pp[:, 0] >= 0) & (pp[:, 0] < SIZE) & (pp[:, 1] >= 0) & (pp[:, 1] < SIZE)
    for x, y in pp[vis]:
        cv2.circle(canvas, (x, y), 1, (70, 70, 70), -1)
    rp = px(P[:, [0, 2]])
    med = np.median(seg)
    for k in range(len(rp) - 1):
        if np.linalg.norm(P[k + 1] - P[k]) < 30 * med:
            cv2.line(canvas, tuple(rp[k]), tuple(rp[k + 1]), (200, 170, 60), 2, cv2.LINE_AA)
    cv2.circle(canvas, tuple(rp[0]), 8, (120, 230, 120), -1)
    cv2.circle(canvas, tuple(rp[-1]), 8, (60, 60, 240), -1)
    return canvas, rp, px, len(rp)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=config.DEFAULT_MAP)
    ap.add_argument("--nats-url", default=f"nats://{config.NATS_HOST}:4222")
    ap.add_argument("--topic", default=config.NATS_TOPIC)
    args = ap.parse_args()

    from .nats_client import NatsClient
    maps = {"rear": build(args.map)}
    try:
        if config.FRONT_MAP != args.map:
            maps["front"] = build(config.FRONT_MAP)
    except Exception as e:
        print(f"[viz] фронт-карта {config.FRONT_MAP} не загружена ({e}); только зад", flush=True)
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
    print("[viz] управление: p=пауза  r=возобновить  c=сброс  Esc=выход", flush=True)

    async def send_control(cmd, **extra):
        await nc.publish(config.NATS_CONTROL_TOPIC,
                         json.dumps({"cmd": cmd, **extra}).encode())
        print(f"[viz] control -> {cmd} {extra}", flush=True)

    col = {"left": (60, 200, 255), "right": (60, 200, 255), "straight": (80, 255, 80),
           "stop": (60, 60, 240), "lost": (90, 90, 240)}
    win = "localization"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    while True:
        c = latest["cmd"] or {}
        canvas, rp, px, n = maps.get(c.get("cam"), maps["rear"])
        img = canvas.copy()
        mt = c.get("move_type")

        if mt and mt != "lost" and c.get("node") is not None:
            node = min(max(c["node"], 0), n - 1)
            tnode = min(max(c.get("target_node", node), 0), n - 1)
            pos = c.get("pos")
            p = tuple(px(pos)[0]) if pos is not None else tuple(rp[node])

            if mt != "stop":
                for a, b in zip(rp[node:tnode], rp[node + 1:tnode + 1]):
                    cv2.line(img, tuple(a), tuple(b), (0, 255, 0), 3, cv2.LINE_AA)

            cv2.line(img, p, tuple(rp[node]), (90, 90, 90), 1, cv2.LINE_AA)
            cv2.circle(img, tuple(rp[node]), 7, (120, 255, 120), 2)
            cv2.circle(img, tuple(rp[tnode]), 7, (0, 235, 235), -1)
            cv2.line(img, p, tuple(rp[tnode]), (0, 235, 235), 1, cv2.LINE_AA)
            if c.get("head") is not None:
                hp = px([pos[0] + c["head"][0] * 0.5, pos[1] + c["head"][1] * 0.5])[0]
                cv2.arrowedLine(img, p, tuple(hp), (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.3)
            cv2.circle(img, p, 9, (60, 60, 255), -1)
            cv2.circle(img, p, 9, (255, 255, 255), 2)

            dm = c.get("dist_to_route_m")
            dtxt = f"{dm:.2f}m" if dm is not None else f"{c.get('dist_to_route', 0):.2f}"
            cv2.putText(img, f"node {node}/{n}   off {dtxt}   deg {c.get('deg', 0.0):+.0f}",
                        (16, 34), FONT, 0.6, (235, 235, 235), 1, cv2.LINE_AA)
            cv2.putText(img, mt.upper(), (16, 74), FONT, 1.1, col.get(mt, (80, 255, 80)), 2, cv2.LINE_AA)
        else:
            cv2.putText(img, "LOST", (16, 74), FONT, 1.1, col["lost"], 2, cv2.LINE_AA)

        if c.get("paused"):
            cv2.putText(img, "PAUSED", (SIZE - 190, 34), FONT, 0.8, (60, 200, 255), 2, cv2.LINE_AA)

        mode = c.get("mode")
        if mode:
            camtxt = f"{mode.upper()}" + (f" [{c.get('cam','?')}]" if mode == "dual" else "")
            ccol = (80, 255, 80) if c.get("cam") == "front" else (60, 200, 255)
            cv2.putText(img, camtxt, (SIZE - 260, 74), FONT, 0.7, ccol, 2, cv2.LINE_AA)
        cv2.putText(img, "1:rear  2:dual", (SIZE - 260, SIZE - 16), FONT, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

        cv2.imshow(win, img)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            break
        elif key == ord("p"):
            await send_control("pause")
        elif key == ord("r"):
            await send_control("resume")
        elif key == ord("c"):
            await send_control("reset")
        elif key == ord("1"):
            await send_control("set_mode", mode="rear")
        elif key == ord("2"):
            await send_control("set_mode", mode="dual")
        await asyncio.sleep(0.001)

    await nc.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main())
