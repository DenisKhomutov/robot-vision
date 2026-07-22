import argparse
import sqlite3
import time
from pathlib import Path

import cv2
import numpy as np
import pycolmap
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MAX_KPTS = 4096


def read_sift(db, image_id):
    r, c, d = db.execute("SELECT rows, cols, data FROM keypoints WHERE image_id=?", (image_id,)).fetchone()
    kp = np.array(np.frombuffer(d, np.float32).reshape(r, c))
    r2, c2, d2 = db.execute("SELECT rows, cols, data FROM descriptors WHERE image_id=?", (image_id,)).fetchone()
    de = np.array(np.frombuffer(d2, np.uint8).reshape(r2, c2)).astype(np.float32)
    de /= np.linalg.norm(de, axis=1, keepdims=True) + 1e-9
    return kp[:, :2].copy(), de


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="map_4f_adaptive_v2")
    ap.add_argument("--images", default="frames_4f_adaptive_v2")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--queries", type=int, default=12)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    work = ROOT / "maps" / args.map
    imdir = ROOT / "data" / args.images
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    rec = pycolmap.Reconstruction(str(work / "sparse" / "0"))
    db = sqlite3.connect(str(work / "database.db"))
    ids = {n: i for i, n in db.execute("SELECT image_id, name FROM images")}
    by_name = {im.name: im for im in rec.images.values()}
    names = sorted(by_name)
    cam = list(rec.cameras.values())[0]
    print(f"карта: {len(names)} кадров, устройство {dev}")

    from torchvision import transforms
    tf = transforms.Compose([transforms.Resize(384), transforms.CenterCrop(336), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                           pretrained=True, trust_repo=True, skip_validation=True).eval().to(dev)

    @torch.inference_mode()
    def embed(path):
        t = tf(Image.open(path).convert("RGB")).unsqueeze(0).to(dev)
        f = model.forward_features(t)
        v = torch.cat([f["x_norm_clstoken"], f["x_norm_patchtokens"].mean(1)], -1)[0]
        return (v / v.norm()).cpu().numpy()

    t0 = time.perf_counter()
    G = np.stack([embed(imdir / n) for n in names])
    print(f"база глобальных дескрипторов: {time.perf_counter()-t0:.0f}с (делается один раз офлайн)")

    mf = {n: read_sift(db, ids[n]) for n in names}
    sift = cv2.SIFT_create(nfeatures=MAX_KPTS)
    idx_of = {n: i for i, n in enumerate(names)}

    step = max(1, len(names) // args.queries)
    T = {"embed": [], "search": [], "sift": [], "match": [], "pnp": [], "total": []}
    ok = wrong = 0
    ranks, inl_true, inl_false = [], [], []

    for qn in names[::step][:args.queries]:
        qi = idx_of[qn]
        t_all = time.perf_counter()

        t = time.perf_counter(); q = embed(imdir / qn); T["embed"].append((time.perf_counter()-t)*1000)
        t = time.perf_counter()
        sim = G @ q
        sim[qi] = -1                       # сам кадр исключаем — иначе тест нечестный
        cand = np.argsort(-sim)[:args.topk]
        T["search"].append((time.perf_counter()-t)*1000)
        ranks.append(int(np.argmin(np.abs(cand - qi))) if np.any(np.abs(cand - qi) <= 3) else -1)

        t = time.perf_counter()
        img = cv2.imread(str(imdir / qn), cv2.IMREAD_GRAYSCALE)
        kp, desc = sift.detectAndCompute(img, None)
        qk = np.array([p.pt for p in kp], np.float32)
        qd = desc.astype(np.float32); qd /= np.linalg.norm(qd, axis=1, keepdims=True) + 1e-9
        qdt = torch.from_numpy(qd).to(dev)
        T["sift"].append((time.perf_counter()-t)*1000)

        t = time.perf_counter()
        p2d, p3d = [], []
        for ci in cand:
            mk, md = mf[names[ci]]
            with torch.inference_mode():
                s = qdt @ torch.from_numpy(md).to(dev).T
                n12 = s.argmax(1); n21 = s.argmax(0)
                mask = (n21[n12] == torch.arange(len(n12), device=dev)) & (s.max(1).values > 0.80)
            m = torch.stack([torch.arange(len(n12), device=dev)[mask], n12[mask]], 1).cpu().numpy()
            pts2 = by_name[names[ci]].points2D
            for a, b in m:
                if b < len(pts2) and pts2[b].has_point3D():
                    p2d.append(qk[a]); p3d.append(rec.points3D[pts2[b].point3D_id].xyz)
        T["match"].append((time.perf_counter()-t)*1000)

        t = time.perf_counter()
        res = pycolmap.estimate_and_refine_absolute_pose(np.array(p2d), np.array(p3d), cam) if len(p2d) >= 6 else None
        T["pnp"].append((time.perf_counter()-t)*1000)
        T["total"].append((time.perf_counter()-t_all)*1000)

        if res:
            C = -res["cam_from_world"].rotation.matrix().T @ res["cam_from_world"].translation
            true_im = by_name[qn]
            Ct = -true_im.cam_from_world().rotation.matrix().T @ true_im.cam_from_world().translation
            err = np.linalg.norm(C - Ct)
            if err < 0.1:
                ok += 1; inl_true.append(res["num_inliers"])
            else:
                wrong += 1; inl_false.append(res["num_inliers"])
                print(f"   {qn}: поза найдена, но ошибка {err:.2f} ед., инлайеров {res['num_inliers']}")

    print(f"\nхолодный старт: верно {ok}/{args.queries}, неверно {wrong}")
    print(f"нужный кадр попал в top-{args.topk}: {sum(1 for x in ranks if x >= 0)}/{len(ranks)}")
    if inl_true: print(f"инлайеров при верной позе: медиана {int(np.median(inl_true))}")
    if inl_false: print(f"инлайеров при НЕверной:    медиана {int(np.median(inl_false))}")
    print(f"\n{'стадия':<10}{'медиана':>10}")
    for k in ("embed", "search", "sift", "match", "pnp", "total"):
        print(f"{k:<10}{np.median(T[k]):>9.0f}мс")


if __name__ == "__main__":
    main()
