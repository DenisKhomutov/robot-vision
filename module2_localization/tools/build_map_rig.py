import argparse
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pycolmap
import torch

ROOT = Path(__file__).resolve().parents[1]
MAX_IMAGE_ID = 2147483647
MIN_MATCHES = 15


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def extract_synced(videos, imdir, step, width, max_frame=None):
    if imdir.exists():
        shutil.rmtree(imdir)
    imdir.mkdir(parents=True)
    counts = []
    for cam, v in enumerate(videos, 1):
        cap = cv2.VideoCapture(str(v))
        if not cap.isOpened():
            log(f"не открывается {v}")
            return None
        idx = kept = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frame and idx > max_frame:
                break
            if idx % step == 0:
                h, w = frame.shape[:2]
                fr = cv2.resize(frame, (width, int(h * width / w)))
                cv2.imwrite(str(imdir / f"{idx:05d}_c{cam}.jpg"), fr,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                kept += 1
            idx += 1
        cap.release()
        counts.append(kept)
        log(f"camera-{cam}: {kept} кадров")
    return counts


def parse(name):
    t, c = name[:-4].split("_c")
    return int(t), int(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", required=True, help="видео камер рига (cam1 cam2 ...)")
    ap.add_argument("--tag", default="map_rig")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--step", type=int, default=6, help="каждый N-й кадр видео")
    ap.add_argument("--overlap", type=int, default=6, help="последовательные пары внутри камеры, ±N шагов")
    ap.add_argument("--kpts", type=int, default=4096)
    ap.add_argument("--det-threshold", type=float, default=0.2)
    ap.add_argument("--max-frame", type=int, default=None, help="обрезать кадры видео после этого индекса")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT / "core"))
    from hub import use_local_weights
    from lightglue import ALIKED, LightGlue
    from lightglue.utils import load_image
    use_local_weights()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    videos = [(ROOT / v if not Path(v).is_absolute() else Path(v)) for v in args.videos]
    imdir = ROOT / "data" / f"frames_{args.tag}"
    work = ROOT / "maps" / args.tag
    work.mkdir(parents=True, exist_ok=True)
    db_path = work / "database.db"
    if db_path.exists():
        db_path.unlink()

    log("--- нарезка синхронных кадров")
    if extract_synced(videos, imdir, args.step, args.width, args.max_frame) is None:
        return 1
    names = sorted(p.name for p in imdir.glob("*.jpg"))
    meta = {n: parse(n) for n in names}
    times = sorted({t for t, _ in meta.values()})
    tstep = times[1] - times[0] if len(times) > 1 else args.step
    log(f"--- кадров {len(names)}, моментов {len(times)}")

    idx = {n: i for i, n in enumerate(names)}
    by_tc = {(t, c): n for n, (t, c) in meta.items()}
    pairs = set()
    for n, (t, c) in meta.items():
        for dt in range(1, args.overlap + 1):
            m = by_tc.get((t + dt * tstep, c))
            if m:
                pairs.add(tuple(sorted((n, m))))
        for dc in (1, -1):
            for dt in (0, tstep, -tstep):
                m = by_tc.get((t + dt, c + dc))
                if m:
                    pairs.add(tuple(sorted((n, m))))
    pairs = sorted(pairs)
    log(f"--- пар {len(pairs)}")

    ext = ALIKED(max_num_keypoints=args.kpts, detection_threshold=args.det_threshold).eval().to(dev)
    lg = LightGlue(features="aliked").eval().to(dev)

    log("--- ALIKED извлечение")
    feats = {}
    h = w = None
    for k, n in enumerate(names):
        with torch.inference_mode():
            im = load_image(str(imdir / n)).to(dev)
            feats[n] = ext.extract(im)
        if h is None:
            h, w = int(im.shape[-2]), int(im.shape[-1])
        if (k + 1) % 100 == 0:
            log(f"  {k+1}/{len(names)}")

    log("--- база + ОТДЕЛЬНАЯ камера OPENCV на каждый _cN (разные объективы)")
    subprocess.run(["colmap", "database_creator", "--database_path", str(db_path)],
                   stdout=subprocess.DEVNULL, check=True)
    db = sqlite3.connect(str(db_path))
    f0 = 1.2 * max(w, h)
    params = np.array([f0, f0, w / 2, h / 2, 0, 0, 0, 0], np.float64).tobytes()
    ncam = max(c for _, c in meta.values())
    for cam in range(1, ncam + 1):
        db.execute("INSERT INTO cameras VALUES (?,?,?,?,?,?)", (cam, 4, w, h, params, 0))
    for k, n in enumerate(names, 1):
        db.execute("INSERT INTO images (image_id, name, camera_id) VALUES (?,?,?)", (k, n, meta[n][1]))
        kp = feats[n]["keypoints"][0].cpu().numpy().astype(np.float32)
        db.execute("INSERT INTO keypoints VALUES (?,?,?,?)", (k, kp.shape[0], 2, kp.tobytes()))
    db.commit()

    log(f"--- ALIKED+LightGlue: {len(pairs)} пар")
    written = weak = 0
    for c, (a, b) in enumerate(pairs):
        with torch.inference_mode():
            out = lg({"image0": feats[a], "image1": feats[b]})
        m = out["matches"][0].cpu().numpy().astype(np.uint32)
        if len(m) >= MIN_MATCHES:
            i, j = idx[a] + 1, idx[b] + 1
            if i > j:
                i, j = j, i
                m = m[:, ::-1]
            m = np.ascontiguousarray(m, np.uint32)
            db.execute("INSERT OR REPLACE INTO matches VALUES (?,?,?,?)",
                       (i * MAX_IMAGE_ID + j, m.shape[0], 2, m.tobytes()))
            written += 1
        else:
            weak += 1
        if (c + 1) % 500 == 0:
            log(f"  {c+1}/{len(pairs)} зап.{written} слаб.{weak}")
    db.commit()
    db.close()
    log(f"--- матчей {written}, слабых {weak}")

    pt = work / "pairs.txt"
    pt.write_text("".join(f"{a} {b}\n" for a, b in pairs))
    subprocess.run(["colmap", "matches_importer", "--database_path", str(db_path),
                    "--match_list_path", str(pt), "--match_type", "pairs"],
                   stdout=subprocess.DEVNULL, check=True)

    sparse = work / "sparse"
    sparse.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["glomap", "mapper", "--database_path", str(db_path),
                        "--image_path", str(imdir), "--output_path", str(sparse)])
    if r.returncode != 0:
        log("glomap упал")
        return 1
    models = sorted(p for p in sparse.iterdir() if p.is_dir())
    if not models:
        log("моделей нет")
        return 1
    best = max(models, key=lambda m: len(pycolmap.Reconstruction(str(m)).images))
    if best.name != "0":
        if (sparse / "0").exists():
            shutil.rmtree(sparse / "0")
        best.rename(sparse / "0")
    sparse0 = sparse / "0"

    rec = pycolmap.Reconstruction(str(sparse0))
    dcache = {n: feats[n]["descriptors"][0].cpu().numpy() for n in names}
    name_of = {im.image_id: im.name for im in rec.images.values()}
    desc_bank, owner, xyz = [], [], []
    for pi, (pid, p3) in enumerate(rec.points3D.items()):
        xyz.append(p3.xyz)
        for el in p3.track.elements[:12]:
            arr = dcache[name_of[el.image_id]]
            if el.point2D_idx < len(arr):
                desc_bank.append(arr[el.point2D_idx])
                owner.append(pi)
    np.savez_compressed(work / "aliked_bank.npz",
                        desc=np.stack(desc_bank).astype(np.float16),
                        owner=np.array(owner, np.int32), xyz=np.stack(xyz))
    reg = [0, 0, 0]
    for im in rec.images.values():
        reg[parse(im.name)[1] - 1] += 1
    log(f"--- банк: {len(xyz)} точек, {len(desc_bank)} дескр., камеры зарегистр. "
        f"c1={reg[0]} c2={reg[1]} c3={reg[2]}, {rec.compute_mean_reprojection_error():.2f}px")
    subprocess.run(["colmap", "model_analyzer", "--path", str(sparse0)])
    return 0


if __name__ == "__main__":
    sys.exit(main())
