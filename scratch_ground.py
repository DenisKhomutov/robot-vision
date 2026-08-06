"""Замер ориентации лидара по плоскости пола (roll вокруг оси "вперёд").

Физика: лидар вертикально, макушка (ось Z лидара) = вперёд по ходу робота.
Значит вперёд = +Z. Влево/вверх лежат в плоскости X-Y лидара, но повёрнуты
на неизвестный roll. Пол горизонтален в мире -> его нормаль даёт "вверх" ->
из неё считаем roll. Принимает облако с того же TCP-сервера (порт 9099).

    uv run --no-sync python scratch_ground.py --host 192.168.40.48
"""
import argparse
import socket
import struct

import numpy as np


def recvall(sock, n):
    buf = bytearray()
    while len(buf) < n:
        c = sock.recv(n - len(buf))
        if not c:
            return None
        buf.extend(c)
    return bytes(buf)


def one_cloud(host, port=9099):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    hdr = recvall(s, 4)
    (n,) = struct.unpack("<I", hdr)
    data = recvall(s, n)
    s.close()
    return np.frombuffer(data, np.float32).reshape(-1, 3).astype(np.float64)


def fit_plane(pts, iters=300, thr=0.05):
    """RANSAC плоскость. Возвращает нормаль (единичную) и число инлайеров."""
    best_n, best_in = None, 0
    rng = np.random.default_rng(0)
    for _ in range(iters):
        idx = rng.choice(len(pts), 3, replace=False)
        p = pts[idx]
        nrm = np.cross(p[1] - p[0], p[2] - p[0])
        ln = np.linalg.norm(nrm)
        if ln < 1e-6:
            continue
        nrm /= ln
        d = np.abs((pts - p[0]) @ nrm)
        cnt = int((d < thr).sum())
        if cnt > best_in:
            best_in, best_n = cnt, nrm
    if best_n[2] < 0:   # нормаль ориентируем последовательно
        best_n = -best_n
    return best_n, best_in


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="192.168.40.48")
    args = ap.parse_args()

    pts = one_cloud(args.host)
    print(f"точек: {len(pts)}")
    print(f"диапазоны:  X[{pts[:,0].min():+.2f},{pts[:,0].max():+.2f}]  "
          f"Y[{pts[:,1].min():+.2f},{pts[:,1].max():+.2f}]  "
          f"Z[{pts[:,2].min():+.2f},{pts[:,2].max():+.2f}]")

    # ищем пол среди точек НЕ прямо по курсу (уберём дальний перёд, оставим низ вокруг)
    n, inl = fit_plane(pts)
    print(f"\nплоскость земли (RANSAC): нормаль = [{n[0]:+.3f}, {n[1]:+.3f}, {n[2]:+.3f}]  "
          f"инлайеров {inl}/{len(pts)} ({100*inl/len(pts):.0f}%)")

    # вперёд считаем +Z лидара (макушка). Нормаль пола = "вверх" мира.
    fwd = np.array([0, 0, 1.0])
    up = n.copy()
    # угол между нормалью пола и осями X/Y лидара покажет roll
    roll = np.degrees(np.arctan2(up[0], up[1]))   # закат вокруг оси вперёд(Z)
    tilt_fwd = np.degrees(np.arcsin(np.clip(up @ fwd, -1, 1)))  # завал нормали в сторону вперёд
    print(f"\nнормаль пола относительно осей лидара:")
    print(f"  roll (закат вокруг 'вперёд'):     {roll:+.1f} deg")
    print(f"  наклон нормали к оси 'вперёд':     {tilt_fwd:+.1f} deg  (0 = лидар строго вертикально)")
    print(f"\nесли roll != 0 -> облако завалено, надо повернуть на -roll вокруг оси Z (вперёд).")


if __name__ == "__main__":
    main()
