import argparse
import sqlite3
import time
from pathlib import Path

import cv2
import numpy as np
import pycolmap
import torch
from kornia.feature.lightglue import LightGlue

ROOT = Path(__file__).resolve().parents[1]
MAX_KPTS = 4096


def read_sift(db, image_id):
    r, c, d = db.execute("SELECT rows, cols, data FROM keypoints WHERE image_id=?", (image_id,)).fetchone()
    kp = np.array(np.frombuffer(d, np.float32).reshape(r, c))
    r2, c2, d2 = db.execute("SELECT rows, cols, data FROM descriptors WHERE image_id=?", (image_id,)).fetchone()
    de = np.array(np.frombuffer(d2, np.uint8).reshape(r2, c2)).astype(np.float32)
    de /= np.linalg.norm(de, axis=1, keepdims=True) + 1e-9
    a11, a12, a21, a22 = kp[:, 2], kp[:, 3], kp[:, 4], kp[:, 5]
    return (kp[:, :2].copy(), np.sqrt(np.abs(a11 * a22 - a12 * a21)).astype(np.float32) + 1e-6,
            np.arctan2(a21, a11).astype(np.float32), de)


def pack(k, s, o, d, size, dev):
    return {"keypoints": torch.from_numpy(k)[None].to(dev), "descriptors": torch.from_numpy(d)[None].to(dev),
            "scales": torch.from_numpy(s)[None].to(dev), "oris": torch.from_numpy(o)[None].to(dev),
            "image_size": size}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="map_4f_adaptive_v2")
    ap.add_argument("--images", default="frames_4f_adaptive_v2")
    ap.add_argument("--topk", type=int, default=5, help="сколько кандидатов даёт retrieval")
    ap.add_argument("--queries", type=int, default=20)
    ap.add_argument("--device", default=None)
    ap.add_argument("--matcher", default="lightglue", choices=["lightglue", "mnn"])
    args = ap.parse_args()

    work = ROOT / "maps" / args.map
    imdir = ROOT / "data" / args.images
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"устройство: {dev}, кандидатов на запрос: {args.topk}")

    rec = pycolmap.Reconstruction(str(work / "sparse" / "0"))
    db = sqlite3.connect(str(work / "database.db"))
    ids = {n: i for i, n in db.execute("SELECT image_id, name FROM images")}
    by_name = {im.name: im for im in rec.images.values()}
    names = sorted(by_name)
    cam = list(rec.cameras.values())[0]
    size = torch.tensor([[float(cam.width), float(cam.height)]]).to(dev)

    sift = cv2.SIFT_create(nfeatures=MAX_KPTS)
    lg = LightGlue(features="sift").eval().to(dev)

    # предзагрузка признаков карты (на роботе это делается один раз при старте)
    t0 = time.perf_counter()
    mfeat = {}
    for n in names:
        mfeat[n] = read_sift(db, ids[n])
    print(f"признаки карты в память: {time.perf_counter()-t0:.1f}с, {len(names)} кадров")

    step = max(1, len(names) // args.queries)
    qnames = names[::step][:args.queries]
    T = {"sift": [], "match": [], "pnp": [], "total": []}
    inl, ok = [], 0

    for qn in qnames:
        img = cv2.imread(str(imdir / qn), cv2.IMREAD_GRAYSCALE)
        t_all = time.perf_counter()

        t = time.perf_counter()
        kp, desc = sift.detectAndCompute(img, None)
        qk = np.array([p.pt for p in kp], np.float32)
        qs = np.array([p.size for p in kp], np.float32)
        qo = np.radians(np.array([p.angle for p in kp], np.float32))
        qd = desc.astype(np.float32)
        qd /= np.linalg.norm(qd, axis=1, keepdims=True) + 1e-9
        T["sift"].append((time.perf_counter() - t) * 1000)

        # retrieval здесь имитируем: берём соседей по маршруту (на роботе это Qdrant/SALAD)
        i = names.index(qn)
        cand = [names[j] for j in range(max(0, i - args.topk // 2), min(len(names), i + args.topk // 2 + 1))
                if names[j] != qn][:args.topk]

        t = time.perf_counter()
        p2d, p3d = [], []
        q = pack(qk, qs, qo, qd, size, dev) if args.matcher == "lightglue" else None
        qdt = torch.from_numpy(qd).to(dev)
        for cn in cand:
            mk, ms, mo, md = mfeat[cn]
            if args.matcher == "lightglue":
                with torch.inference_mode():
                    out = lg({"image0": q, "image1": pack(mk, ms, mo, md, size, dev)})
                m = out["matches"][0].cpu().numpy()
            else:
                with torch.inference_mode():
                    sim = qdt @ torch.from_numpy(md).to(dev).T
                    n12 = sim.argmax(1); n21 = sim.argmax(0)
                    mask = n21[n12] == torch.arange(len(n12), device=dev)
                    mask &= sim.max(1).values > 0.80
                m = torch.stack([torch.arange(len(n12), device=dev)[mask], n12[mask]], 1).cpu().numpy()
            if not len(m):
                continue
            pts2 = by_name[cn].points2D
            for a, b in m:
                if b < len(pts2) and pts2[b].has_point3D():
                    p2d.append(qk[a])
                    p3d.append(rec.points3D[pts2[b].point3D_id].xyz)
        T["match"].append((time.perf_counter() - t) * 1000)

        t = time.perf_counter()
        res = None
        if len(p2d) >= 6:
            res = pycolmap.estimate_and_refine_absolute_pose(np.array(p2d), np.array(p3d), cam)
        T["pnp"].append((time.perf_counter() - t) * 1000)
        T["total"].append((time.perf_counter() - t_all) * 1000)
        if res:
            ok += 1
            inl.append(res["num_inliers"])

    print(f"\nлокализовано: {ok}/{len(qnames)}, инлайеров медиана {int(np.median(inl)) if inl else 0}")
    print(f"{'стадия':<12}{'медиана':>10}{'95%':>10}")
    for k in ("sift", "match", "pnp", "total"):
        v = np.array(T[k])
        print(f"{k:<12}{np.median(v):>9.0f}мс{np.percentile(v,95):>9.0f}мс")
    print(f"\nчастота: {1000/np.median(T['total']):.1f} Гц на {dev}")


if __name__ == "__main__":
    main()
