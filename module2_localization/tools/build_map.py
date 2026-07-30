import argparse
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pycolmap
import torch

ROOT = Path(__file__).resolve().parents[1]
MAX_IMAGE_ID = 2147483647
MIN_MATCHES = 10
VOCAB_TREE = ROOT / "data" / "vocab_tree_flickr100K_words32K.bin"
VOCAB_URL = "https://demuc.de/colmap/vocab_tree_flickr100K_words32K.bin"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def run(cmd, tag, quiet=True):
    log(f"--- {tag}")
    t0 = time.perf_counter()
    p = subprocess.run(cmd, stdout=subprocess.DEVNULL if quiet else None)
    if p.returncode != 0:
        log(f"ОШИБКА '{tag}' код {p.returncode}")
        return False
    log(f"--- {tag}: {time.perf_counter() - t0:.0f}с")
    return True


def fetch_vocab():
    if VOCAB_TREE.exists():
        return True
    VOCAB_TREE.parent.mkdir(parents=True, exist_ok=True)
    log("качаю vocab tree (~200МБ)")
    return subprocess.run(["curl", "-sL", "-o", str(VOCAB_TREE), VOCAB_URL]).returncode == 0


def vocab_pairs(imdir, work, neighbors):
    rdb = work / "retrieval.db"
    if rdb.exists():
        rdb.unlink()
    if not run(["colmap", "feature_extractor", "--database_path", str(rdb),
                "--image_path", str(imdir), "--ImageReader.single_camera", "1",
                "--ImageReader.camera_model", "OPENCV", "--SiftExtraction.use_gpu", "1",
                "--SiftExtraction.max_num_features", "4096"], "SIFT (оракул отбора пар)"):
        return None, None
    if not fetch_vocab():
        log("нет словаря")
        return None, None

    log(f"--- словарь: ретривер, топ-{neighbors} соседей")
    t0 = time.perf_counter()
    p = subprocess.run(["colmap", "vocab_tree_retriever", "--database_path", str(rdb),
                        "--vocab_tree_path", str(VOCAB_TREE),
                        "--num_neighbors", str(neighbors)],
                       capture_output=True, text=True)
    if p.returncode != 0:
        log(f"ретривер упал код {p.returncode}\n{p.stderr[-800:]}")
        return None, None
    pairs, cur = set(), None
    for line in p.stdout.splitlines():
        if "Querying for image" in line:
            cur = line.split("Querying for image", 1)[1].split("[")[0].strip()
        elif "image_name=" in line and cur is not None:
            nb = line.split("image_name=", 1)[1].split(",", 1)[0].strip()
            if nb != cur:
                pairs.add(tuple(sorted((cur, nb))))
    log(f"--- словарь: {len(pairs)} пар за {time.perf_counter() - t0:.0f}с")

    db = sqlite3.connect(str(rdb))
    cam = db.execute("SELECT model, width, height, params FROM cameras LIMIT 1").fetchone()
    db.close()
    return pairs, cam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="frames_office")
    ap.add_argument("--tag", default="map_office_v2")
    ap.add_argument("--kpts", type=int, default=4096, help="ПОТОЛОК точек (не добавляет: ALIKED "
                    "ограничен порогом детекции, ~873 при дефолте). Реальная ручка — --det-threshold")
    ap.add_argument("--det-threshold", type=float, default=0.2, help="порог детекции ALIKED; "
                    "ниже -> больше (более слабых) точек. 0.05 ~= x1.8, 0.02 ~= x2.5")
    ap.add_argument("--overlap", type=int, default=10, help="последовательные пары ±N")
    ap.add_argument("--neighbors", type=int, default=20, help="соседей на кадр в словаре")
    ap.add_argument("--pairs", default="vocab",
                    choices=["vocab", "sequential", "exhaustive", "loop"])
    ap.add_argument("--loop-head", type=int, default=25, help="loop: первых кадров у старта")
    ap.add_argument("--loop-tail", type=int, default=25, help="loop: последних кадров у финиша")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT / "core"))
    from hub import use_local_weights
    from lightglue import ALIKED, LightGlue
    from lightglue.utils import load_image
    use_local_weights()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    imdir = ROOT / "data" / args.images
    names = sorted(p.name for p in imdir.glob("*.jpg"))
    if not names:
        log(f"нет кадров в {imdir}")
        return 1
    work = ROOT / "maps" / args.tag
    work.mkdir(parents=True, exist_ok=True)
    db_path = work / "database.db"
    if db_path.exists():
        db_path.unlink()
    log(f"кадров {len(names)}, точек/кадр {args.kpts}, пары {args.pairs}")

    idx = {n: k for k, n in enumerate(names)}
    seq = {tuple(sorted((names[i], names[j]))) for i in range(len(names))
           for j in range(i + 1, min(i + 1 + args.overlap, len(names)))}
    cam_row = None
    if args.pairs == "vocab":
        loops, cam_row = vocab_pairs(imdir, work, args.neighbors)
        if loops is None:
            return 1
        pairs = sorted(seq | loops)
        log(f"пар всего {len(pairs)} (послед. {len(seq)}, петлевых доп. {len(pairs) - len(seq)})")
    elif args.pairs == "exhaustive":
        pairs = [tuple(sorted((names[i], names[j]))) for i in range(len(names))
                 for j in range(i + 1, len(names))]
        log(f"пар всего {len(pairs)} (exhaustive)")
    elif args.pairs == "loop":
        h, t = min(args.loop_head, len(names)), min(args.loop_tail, len(names))
        loops = {tuple(sorted((names[i], names[len(names) - 1 - j])))
                 for i in range(h) for j in range(t)}
        loops -= seq
        pairs = sorted(seq | loops)
        log(f"пар всего {len(pairs)} (послед. {len(seq)}, петлевых {len(loops)})")
    else:
        pairs = sorted(seq)
        log(f"пар всего {len(pairs)} (последовательные)")

    ext = ALIKED(max_num_keypoints=args.kpts, detection_threshold=args.det_threshold).eval().to(dev)
    lg = LightGlue(features="aliked").eval().to(dev)

    log("--- ALIKED: извлечение")
    t0 = time.perf_counter()
    feats = {}
    h = w = None
    for k, n in enumerate(names):
        with torch.inference_mode():
            im = load_image(str(imdir / n)).to(dev)
            feats[n] = ext.extract(im)
        if h is None:
            h, w = int(im.shape[-2]), int(im.shape[-1])
        if (k + 1) % 100 == 0:
            el = time.perf_counter() - t0
            log(f"  {k+1}/{len(names)} | {el:.0f}с | ~{el/(k+1)*(len(names)-k-1):.0f}с")
    med = int(np.median([feats[n]["keypoints"].shape[1] for n in names]))
    log(f"--- точек: медиана {med}")

    log("--- база (colmap database_creator)")
    subprocess.run(["colmap", "database_creator", "--database_path", str(db_path)],
                   stdout=subprocess.DEVNULL, check=True)
    db = sqlite3.connect(str(db_path))
    if cam_row is not None:
        model, cw, ch, params = cam_row
        db.execute("INSERT INTO cameras VALUES (?,?,?,?,?,?)", (1, model, cw, ch, params, 0))
    else:
        f0 = 1.2 * max(w, h)
        params = np.array([f0, f0, w / 2, h / 2, 0, 0, 0, 0], np.float64).tobytes()
        db.execute("INSERT INTO cameras VALUES (?,?,?,?,?,?)", (1, 4, w, h, params, 0))
    for k, n in enumerate(names, 1):
        db.execute("INSERT INTO images (image_id, name, camera_id) VALUES (?,?,?)", (k, n, 1))
        kp = feats[n]["keypoints"][0].cpu().numpy().astype(np.float32)
        db.execute("INSERT INTO keypoints VALUES (?,?,?,?)", (k, kp.shape[0], 2, kp.tobytes()))
    db.commit()

    log(f"--- ALIKED+LightGlue: {len(pairs)} пар")
    t0 = time.perf_counter()
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
            el = time.perf_counter() - t0
            log(f"  {c+1}/{len(pairs)} | зап.{written} слаб.{weak} | ~{el/(c+1)*(len(pairs)-c-1):.0f}с")
    db.commit()
    db.close()
    log(f"--- матчей записано {written}, слабых {weak}")

    pt = work / "pairs.txt"
    pt.write_text("".join(f"{a} {b}\n" for a, b in pairs))
    if not run(["colmap", "matches_importer", "--database_path", str(db_path),
                "--match_list_path", str(pt), "--match_type", "pairs"], "геом. проверка"):
        return 1

    sparse = work / "sparse"
    sparse.mkdir(parents=True, exist_ok=True)
    if not run(["glomap", "mapper", "--database_path", str(db_path),
                "--image_path", str(imdir), "--output_path", str(sparse)], "glomap (SfM с нуля)",
               quiet=False):
        return 1

    models = sorted(p for p in sparse.iterdir() if p.is_dir())
    if not models:
        log("glomap не собрал модель")
        return 1
    best = max(models, key=lambda m: len(pycolmap.Reconstruction(str(m)).images))
    if best.name != "0":
        if (sparse / "0").exists():
            shutil.rmtree(sparse / "0")
        best.rename(sparse / "0")
    sparse0 = sparse / "0"
    log(f"моделей {len(models)}, взята {best.name} -> 0")

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
    log(f"--- банк: {len(xyz)} точек, {len(desc_bank)} дескрипторов, "
        f"{len(rec.images)}/{len(names)} камер, {rec.compute_mean_reprojection_error():.2f}px")
    subprocess.run(["colmap", "model_analyzer", "--path", str(sparse0)])
    return 0


if __name__ == "__main__":
    sys.exit(main())
