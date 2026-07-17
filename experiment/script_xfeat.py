import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "xfeat_src"))
from modules.xfeat import XFeat  # noqa: E402

INTRINSICS = "experiment/camera_intrinsics.npz"
XFEAT_W = "experiment/weights/xfeat.pt"
OUT_DIR = Path("experiment/pose_out")

ROUTE = "data/route_reference/route_B/forward/front"
ROUTE_PAIRS = [(1, 2), (2, 3), (5, 6), (9, 10), (9, 11), (14, 15), (17, 18)]

TOP_K = 4096
MIN_MATCHES = 20
RANSAC_PX = 3.0
AGREE_TOL = 3.0


def rot_to_yaw(r):
    return -float(np.degrees(np.arctan2(r[0, 2], r[2, 2])))


def yaw_from_homography(p0, p1, k):
    h, mask = cv2.findHomography(p0, p1, cv2.USAC_MAGSAC, RANSAC_PX, confidence=0.999, maxIters=10000)
    if h is None:
        return None, 0
    r = np.linalg.inv(k) @ h @ k
    u, _, vt = np.linalg.svd(r)
    r = u @ vt
    if np.linalg.det(r) < 0:
        r = u @ np.diag([1.0, 1.0, -1.0]) @ vt
    return rot_to_yaw(r), int(mask.sum()) if mask is not None else 0


def yaw_from_essential(p0, p1, k):
    e, mask = cv2.findEssentialMat(p0, p1, k, method=cv2.USAC_MAGSAC, prob=0.999, threshold=1.0)
    if e is None or e.shape[0] % 3 != 0:
        return None, 0
    n, r, _, _ = cv2.recoverPose(e[:3], p0, p1, k, mask=mask)
    return rot_to_yaw(r), int(n)


def load_model():
    return XFeat(weights=XFEAT_W, top_k=TOP_K)


def match(xf, i0, i1):
    t = time.perf_counter()
    p0, p1 = xf.match_xfeat(i0, i1, top_k=TOP_K)
    return p0, p1, (time.perf_counter() - t) * 1000


def estimate(p0, p1, k):
    out = {"yaw_deg": None, "homog": None, "essent": None, "disagree_deg": None,
           "confident": False, "n_h": 0, "n_e": 0}
    if len(p0) < MIN_MATCHES:
        return out
    yh, nh = yaw_from_homography(p0, p1, k)
    ye, ne = yaw_from_essential(p0, p1, k)
    out.update({"homog": yh, "essent": ye, "n_h": nh, "n_e": ne})
    if yh is None and ye is None:
        return out
    if yh is None or ye is None:
        out["yaw_deg"] = yh if yh is not None else ye
        return out
    gap = abs(yh - ye)
    out["disagree_deg"] = round(gap, 2)
    out["confident"] = gap <= AGREE_TOL
    out["yaw_deg"] = round((yh + ye) / 2 if gap <= AGREE_TOL else yh, 2)
    return out


def fmt(v):
    return f"{v:+7.2f}" if v is not None else "   n/a "


def draw_pair(i0, i1, p0, p1, title, res, max_draw=60):
    h0, w0 = i0.shape[:2]
    h1, w1 = i1.shape[:2]
    top = np.zeros((max(h0, h1), w0 + w1, 3), np.uint8)
    top[:h0, :w0] = i0
    top[:h1, w0:w0 + w1] = i1

    inl = np.zeros(len(p0), bool)
    if len(p0) >= MIN_MATCHES:
        _, mask = cv2.findHomography(p0, p1, cv2.USAC_MAGSAC, RANSAC_PX, confidence=0.999, maxIters=10000)
        if mask is not None:
            inl = mask.ravel().astype(bool)

    step = max(1, len(p0) // max_draw)
    for i in range(0, len(p0), step):
        a = (int(p0[i][0]), int(p0[i][1]))
        b = (int(p1[i][0]) + w0, int(p1[i][1]))
        col = (120, 230, 120) if inl[i] else (60, 60, 160)
        cv2.line(top, a, b, col, 1, cv2.LINE_AA)
        cv2.circle(top, a, 2, col, -1)
        cv2.circle(top, b, 2, col, -1)

    f = cv2.FONT_HERSHEY_SIMPLEX
    bar = np.full((170, top.shape[1], 3), 22, np.uint8)
    cv2.putText(bar, title, (16, 34), f, 0.7, (235, 235, 235), 2)
    cv2.putText(bar, f"XFeat + MNN    matches {len(p0)}   inliers_H {int(inl.sum())}",
                (16, 66), f, 0.55, (150, 150, 150), 1)

    yaw = res["yaw_deg"]
    col = (120, 230, 120) if res["confident"] else (0, 165, 255)
    cv2.putText(bar, "YAW (mean H+E)" if res["confident"] else "YAW (H only, disagree)",
                (16, 98), f, 0.5, (135, 135, 135), 1)
    cv2.putText(bar, f"{yaw:+.2f}" if yaw is not None else "n/a", (16, 140), f, 1.1, col, 2)

    x = 320
    for label, val in (("homography", res["homog"]), ("essential", res["essent"]),
                       ("disagree", res["disagree_deg"])):
        cv2.putText(bar, label, (x, 98), f, 0.5, (135, 135, 135), 1)
        cv2.putText(bar, f"{val:+.2f}" if val is not None else "n/a", (x, 140), f, 0.8, (200, 200, 200), 2)
        x += 220
    return np.vstack([top, bar])


def run_route(xf, k, dist):
    print(f"{'para':<10}{'yaw':>9}{'homog':>9}{'essent':>9}{'disagree':>10}"
          f"{'conf':>6}{'par':>7}{'in_h':>6}{'in_e':>6}{'ms':>7}")
    for a, b in ROUTE_PAIRS:
        f0, f1 = f"{ROUTE}/front{a}.jpg", f"{ROUTE}/front{b}.jpg"
        if not (Path(f0).exists() and Path(f1).exists()):
            continue
        i0 = cv2.undistort(cv2.imread(f0), k, dist)
        i1 = cv2.undistort(cv2.imread(f1), k, dist)
        p0, p1, ms = match(xf, i0, i1)
        r = estimate(p0, p1, k)
        print(f"{f'{a}->{b}':<10}{fmt(r['yaw_deg']):>9}{fmt(r['homog']):>9}{fmt(r['essent']):>9}"
              f"{fmt(r['disagree_deg']):>10}{str(r['confident']):>6}{len(p0):>7}"
              f"{r['n_h']:>6}{r['n_e']:>6}{ms:>7.0f}")
        panel = draw_pair(i0, i1, p0, p1, f"front{a} -> front{b}", r)
        cv2.imwrite(str(OUT_DIR / f"route_{a}_{b}.jpg"),
                    cv2.resize(panel, (panel.shape[1] // 2, panel.shape[0] // 2)))


def main():
    intr = np.load(INTRINSICS)
    k, dist = intr["K"], intr["dist"]
    w = int(intr["image_size"][0]) if "image_size" in intr else 1280
    hfov = float(np.degrees(2 * np.arctan(w / (2 * k[0, 0]))))
    print(f"[INFO] K: fx={k[0, 0]:.1f} cx={k[0, 2]:.1f}   HFOV {hfov:.1f}   AGREE_TOL {AGREE_TOL}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_route(load_model(), k, dist)


if __name__ == "__main__":
    main()
