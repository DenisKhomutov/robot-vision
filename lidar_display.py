import argparse
import socket
import struct
import threading
from collections import deque

import cv2
import numpy as np

PORT = 9099
SIZE = 700
RANGE = 8.0                      
ACCUM = 5
FONT = cv2.FONT_HERSHEY_SIMPLEX

ROLL_DEG, PITCH_DEG, YAW_DEG, MIRROR = 0.0, -180.0, 18.0, False

_scans = deque(maxlen=ACCUM)
_latest = {"xyz": None, "n": 0}
_lock = threading.Lock()


def recvall(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def receiver(host):
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((host, PORT))
            print(f"[lidar] подключился к {host}:{PORT}", flush=True)
            while True:
                hdr = recvall(s, 4)
                if hdr is None:
                    break
                (n,) = struct.unpack("<I", hdr)
                data = recvall(s, n)
                if data is None:
                    break
                xyz = np.frombuffer(data, np.float32).reshape(-1, 3)
                with _lock:
                    _scans.append(xyz)
                    _latest["xyz"] = np.concatenate(_scans, axis=0)
                    _latest["n"] = len(xyz)
        except OSError as e:
            print(f"[lidar] нет связи ({e}); переподключаюсь...", flush=True)
        threading.Event().wait(1.0)


def panel(xyz, a, b, la, lb):
    img = np.full((SIZE, SIZE, 3), 20, np.uint8)
    c = SIZE // 2
    s = (SIZE * 0.45) / RANGE
    cv2.line(img, (c, 0), (c, SIZE), (50, 50, 50), 1)
    cv2.line(img, (0, c), (SIZE, c), (50, 50, 50), 1)
    for r in (1, 2, 4, 8):
        cv2.circle(img, (c, c), int(r * s), (40, 40, 40), 1)
    if xyz is not None and len(xyz):
        u = (c + xyz[:, a] * s).astype(int)
        v = (c - xyz[:, b] * s).astype(int)
        h = xyz[:, 2]
        lo, hi = np.percentile(h, [2, 98])
        t = np.clip((h - lo) / (hi - lo + 1e-9), 0, 1)
        m = (u >= 0) & (u < SIZE) & (v >= 0) & (v < SIZE)
        col = np.stack([(255 * (1 - t)), (255 * (1 - np.abs(t - .5) * 2)), (255 * t)], 1).astype(np.uint8)
        img[v[m], u[m]] = col[m]
    cv2.putText(img, f"{la}->", (SIZE - 60, c - 8), FONT, 0.5, (200, 200, 200), 1)
    cv2.putText(img, f"^{lb}", (c + 6, 20), FONT, 0.5, (200, 200, 200), 1)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="192.168.40.48", help="IP Jetson")
    args = ap.parse_args()

    threading.Thread(target=receiver, args=(args.host,), daemon=True).start()

    st = {"roll": ROLL_DEG, "pitch": PITCH_DEG, "yaw": YAW_DEG, "mirror": MIRROR}
    STEP = 1.0

    def rebuild():
        r, p, y = np.radians([st["roll"], st["pitch"], st["yaw"]])
        cx, sx = np.cos(r), np.sin(r)
        cy_, sy = np.cos(p), np.sin(p)
        cz, sz = np.cos(y), np.sin(y)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy_, 0, sy], [0, 1, 0], [-sy, 0, cy_]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx
        if st["mirror"]:
            R = np.diag([-1.0, 1.0, 1.0]) @ R
        return R

    win = "lidar"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    while True:
        with _lock:
            raw = None if _latest["xyz"] is None else _latest["xyz"].copy()
            n = _latest["n"]
        xyz = None if raw is None else raw @ rebuild().T
        top = panel(xyz, 0, 1, "FWD", "LEFT")
        front = panel(xyz, 0, 2, "FWD", "UP")
        both = np.hstack([top, front])
        cv2.putText(both, f"roll={st['roll']:.0f} pitch={st['pitch']:.0f} yaw={st['yaw']:.0f} "
                    f"mirror={st['mirror']}   points={n}",
                    (10, 24), FONT, 0.55, (120, 220, 255), 1)
        cv2.putText(both, "[ ]=roll  ; '=pitch  , .=yaw  m=mirror  Esc=exit",
                    (10, SIZE - 12), FONT, 0.5, (180, 180, 180), 1)
        cv2.imshow(win, both)
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break
        elif k == ord("["):
            st["roll"] -= STEP
        elif k == ord("]"):
            st["roll"] += STEP
        elif k == ord(";"):
            st["pitch"] -= STEP
        elif k == ord("'"):
            st["pitch"] += STEP
        elif k == ord(","):
            st["yaw"] -= STEP
        elif k == ord("."):
            st["yaw"] += STEP
        elif k == ord("m"):
            st["mirror"] = not st["mirror"]
    print(f"\nИТОГ: roll={st['roll']} pitch={st['pitch']} yaw={st['yaw']} mirror={st['mirror']}",
          flush=True)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
