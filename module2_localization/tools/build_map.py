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
VOCAB_TREE = ROOT / "data" / "vocab_tree_flickr100K_words32K.bin"
VOCAB_URL = "https://demuc.de/colmap/vocab_tree_flickr100K_words32K.bin"


def parse_offsets(value):
    try:
        offsets = sorted({int(item) for item in value.split(",") if item.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("смещения должны быть целыми числами через запятую") from exc
    if not offsets or offsets[0] < 1:
        raise argparse.ArgumentTypeError("смещения должны быть положительными")
    return offsets


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
    ap.add_argument("--pair-offsets", type=parse_offsets, default=None,
                    help="вместо всех пар до overlap использовать смещения, например 1,3,6,10,16,25")
    ap.add_argument("--neighbors", type=int, default=20, help="соседей на кадр в словаре")
    ap.add_argument("--pairs", default="sequential",
                    choices=["vocab", "sequential", "exhaustive", "loop"])
    ap.add_argument("--mapper", choices=["colmap", "glomap"], default="colmap",
                    help="colmap: единая инкрементальная SfM (по умолчанию); glomap оставлен для сравнения")
    ap.add_argument("--init-image-ids", type=int, nargs=2, metavar=("ID1", "ID2"),
                    help="стартовая пара COLMAP; для последовательного маршрута лучше начинать с начала")
    ap.add_argument("--camera-model", choices=["simple-radial", "opencv"],
                    default="simple-radial", help="модель самокалибровки без --calib; SIMPLE_RADIAL "
                    "лучше обусловлена для одной камеры")
    ap.add_argument("--refine-intrinsics", action="store_true",
                    help="разрешить BA менять фокус/дисторсию (для длинного прямого маршрута "
                         "обычно нельзя: возникает схлопывание масштаба)")
    ap.add_argument("--colors", action=argparse.BooleanOptionalAction, default=False,
                    help="добавить RGB-цвета 3D-точкам из исходных изображений; по умолчанию "
                         "сохраняется бесцветная геометрия")
    ap.add_argument("--loop-head", type=int, default=25, help="loop: первых кадров у старта")
    ap.add_argument("--loop-tail", type=int, default=25, help="loop: последних кадров у финиша")
    ap.add_argument("--focal", type=float, default=None,
                    help="стартовый фокус в px (fx=fy). Без него — догадка 1.2*сторона (часто "
                         "плохо уточняется на прямых коридорах -> кривой fx -> ошибка поворотов)")
    ap.add_argument("--calib", default=None,
                    help="npz с калибровкой (K, dist, image_size) -> камера OPENCV, интринсики "
                         "ФИКСИРУЮТСЯ в BA (не уточняются)")
    ap.add_argument("--masks", default=None,
                    help="папка масок (data/<...>): точки внутри маски (255) отбрасываются, "
                         "картинка не трогается — без ложных точек на кромке заливки")
    ap.add_argument("--overwrite", action="store_true", help="явно заменить существующую карту")
    ap.add_argument("--prepare-only", action="store_true",
                    help="подготовить database.db с ALIKED/LightGlue и геометрической проверкой, "
                         "но не запускать mapper")
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
    if Path(args.tag).name != args.tag:
        log("ОШИБКА: --tag должен быть именем каталога внутри module2_localization/maps")
        return 2
    work = ROOT / "maps" / args.tag
    if work.exists():
        if not args.overwrite:
            log(f"ОШИБКА: карта {args.tag} уже существует; задайте другой --tag или --overwrite")
            return 2
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    db_path = work / "database.db"
    log(f"кадров {len(names)}, точек/кадр {args.kpts}, пары {args.pairs}, mapper {args.mapper}")

    idx = {n: k for k, n in enumerate(names)}
    offsets = args.pair_offsets or list(range(1, args.overlap + 1))
    seq = {tuple(sorted((names[i], names[j]))) for i in range(len(names))
           for offset in offsets for j in (i + offset,) if j < len(names)}
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

    maskdir = ROOT / "data" / args.masks if args.masks else None
    if maskdir and not maskdir.exists():
        log(f"нет папки масок {maskdir}")
        return 1

    def drop_masked(f, name, ih, iw):
        mp = maskdir / (Path(name).stem + ".png")
        if not mp.exists():
            return f, 0
        mimg = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if mimg.shape != (ih, iw):
            mimg = cv2.resize(mimg, (iw, ih), interpolation=cv2.INTER_NEAREST)
        kp = f["keypoints"][0]
        x = kp[:, 0].round().long().clamp(0, iw - 1)
        y = kp[:, 1].round().long().clamp(0, ih - 1)
        mt = torch.from_numpy(mimg).to(kp.device)
        keep = mt[y, x] == 0
        dropped = int((~keep).sum())
        for key in ("keypoints", "keypoint_scores", "descriptors"):
            if key in f:
                f[key] = f[key][:, keep]
        return f, dropped

    log("--- ALIKED: извлечение" + (" (+маски)" if maskdir else ""))
    t0 = time.perf_counter()
    feats = {}
    h = w = None
    dropped_total = 0
    for k, n in enumerate(names):
        with torch.inference_mode():
            im = load_image(str(imdir / n)).to(dev)
            feats[n] = ext.extract(im)
        ih, iw = int(im.shape[-2]), int(im.shape[-1])
        if maskdir:
            feats[n], dr = drop_masked(feats[n], n, ih, iw)
            dropped_total += dr
        if h is None:
            h, w = ih, iw
        if (k + 1) % 100 == 0:
            el = time.perf_counter() - t0
            log(f"  {k+1}/{len(names)} | {el:.0f}с | ~{el/(k+1)*(len(names)-k-1):.0f}с")
    med = int(np.median([feats[n]["keypoints"].shape[1] for n in names]))
    log(f"--- точек: медиана {med}" + (f", выброшено по маскам {dropped_total}" if maskdir else ""))

    log("--- база (colmap database_creator)")
    subprocess.run(["colmap", "database_creator", "--database_path", str(db_path)],
                   stdout=subprocess.DEVNULL, check=True)
    db = sqlite3.connect(str(db_path))
    if args.calib:
        cal = np.load(args.calib)
        K = cal["K"]
        dd = cal["dist"].ravel()
        cw, ch = (int(x) for x in cal["image_size"])
        sx, sy = w / cw, h / ch          # пересчёт K, если калибровка снята в другом разрешении
        fx, fy, cxp, cyp = K[0, 0] * sx, K[1, 1] * sy, K[0, 2] * sx, K[1, 2] * sy
        params = np.array([fx, fy, cxp, cyp, dd[0], dd[1], dd[2], dd[3]], np.float64).tobytes()
        db.execute("INSERT INTO cameras VALUES (?,?,?,?,?,?)", (1, 4, w, h, params, 0))
        log(f"--- калибровка: fx={fx:.0f} fy={fy:.0f} cx={cxp:.0f} cy={cyp:.0f} "
            f"k1={dd[0]:.3f} (фиксируется в BA)")
    elif cam_row is not None:
        model, cw, ch, params = cam_row
        db.execute("INSERT INTO cameras VALUES (?,?,?,?,?,?)", (1, model, cw, ch, params, 0))
    else:
        f0 = args.focal if args.focal else 1.2 * max(w, h)
        if args.camera_model == "simple-radial":
            model_id = 2
            params = np.array([f0, w / 2, h / 2, 0], np.float64).tobytes()
            model_name = "SIMPLE_RADIAL"
        else:
            model_id = 4
            params = np.array([f0, f0, w / 2, h / 2, 0, 0, 0, 0], np.float64).tobytes()
            model_name = "OPENCV"
        db.execute("INSERT INTO cameras VALUES (?,?,?,?,?,?)", (1, model_id, w, h, params, 0))
        log(f"--- камера {model_name}, стартовый фокус {f0:.0f}px" +
            (" (задан)" if args.focal else " (догадка 1.2*сторона)"))
    for k, n in enumerate(names, 1):
        db.execute("INSERT INTO images (image_id, name, camera_id) VALUES (?,?,?)", (k, n, 1))
        kp = feats[n]["keypoints"][0].cpu().numpy().astype(np.float32)
        db.execute("INSERT INTO keypoints VALUES (?,?,?,?)", (k, kp.shape[0], 2, kp.tobytes()))
    db.commit()

    log(f"--- ALIKED+LightGlue: {len(pairs)} пар")
    t0 = time.perf_counter()
    written = weak = 0
    written_pairs = []
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
            written_pairs.append((a, b))
        else:
            weak += 1
        if (c + 1) % 500 == 0:
            el = time.perf_counter() - t0
            log(f"  {c+1}/{len(pairs)} | зап.{written} слаб.{weak} | ~{el/(c+1)*(len(pairs)-c-1):.0f}с")
    db.commit()
    db.close()
    log(f"--- матчей записано {written}, слабых {weak}")

    pt = work / "pairs.txt"
    pt.write_text("".join(f"{a} {b}\n" for a, b in written_pairs))
    if not run(["colmap", "matches_importer", "--database_path", str(db_path),
                "--match_list_path", str(pt), "--match_type", "pairs",
                "--SiftMatching.num_threads", "1", "--SiftMatching.use_gpu", "0"],
               "геом. проверка"):
        return 1

    if args.prepare_only:
        log(f"--- база подготовлена: {db_path}")
        log("--- mapper пропущен (--prepare-only)")
        return 0

    sparse = work / "sparse"
    sparse.mkdir(parents=True, exist_ok=True)
    if args.mapper == "colmap":
        mapper_cmd = ["colmap", "mapper", "--database_path", str(db_path),
                      "--image_path", str(imdir), "--output_path", str(sparse),
                      "--Mapper.multiple_models", "0", "--Mapper.extract_colors",
                      "1" if args.colors else "0",
                      "--Mapper.ba_refine_principal_point", "0",
                      "--Mapper.ba_global_images_freq", "100",
                      "--Mapper.ba_global_points_freq", "50000",
                      "--Mapper.ba_global_max_num_iterations", "100",
                      "--Mapper.ba_global_max_refinements", "10",
                      "--Mapper.tri_ignore_two_view_tracks", "0",
                      "--Mapper.tri_min_angle", "1.5",
                      "--Mapper.filter_min_tri_angle", "1.5",
                      "--Mapper.local_ba_min_tri_angle", "6",
                      "--Mapper.abs_pose_min_num_inliers", "15",
                      "--Mapper.abs_pose_min_inlier_ratio", "0.1"]
        if args.calib or not args.refine_intrinsics:
            mapper_cmd += ["--Mapper.ba_refine_focal_length", "0",
                           "--Mapper.ba_refine_extra_params", "0"]
        if args.init_image_ids:
            mapper_cmd += ["--Mapper.init_image_id1", str(args.init_image_ids[0]),
                           "--Mapper.init_image_id2", str(args.init_image_ids[1]),
                           "--Mapper.init_min_tri_angle", "4",
                           "--Mapper.init_max_forward_motion", "1"]
        mapper_label = "COLMAP incremental (единая 2D-3D реконструкция)"
    else:
        mapper_cmd = ["glomap", "mapper", "--database_path", str(db_path),
                      "--image_path", str(imdir), "--output_path", str(sparse)]
        if args.calib or not args.refine_intrinsics:
            mapper_cmd += ["--BundleAdjustment.optimize_intrinsics", "0"]
        mapper_label = "GLOMAP global"
    if not run(mapper_cmd, mapper_label, quiet=False):
        return 1

    models = sorted(p for p in sparse.iterdir() if p.is_dir())
    if not models:
        log(f"{args.mapper} не собрал модель")
        return 1
    best = max(models, key=lambda m: len(pycolmap.Reconstruction(str(m)).images))
    if best.name != "0":
        if (sparse / "0").exists():
            shutil.rmtree(sparse / "0")
        best.rename(sparse / "0")
    sparse0 = sparse / "0"
    log(f"моделей {len(models)}, взята {best.name} -> 0")

    rec = pycolmap.Reconstruction(str(sparse0))
    min_registered = max(2, int(0.9 * len(names)))
    if len(rec.images) < min_registered:
        log(f"ОШИБКА: зарегистрировано только {len(rec.images)}/{len(names)} камер "
            f"(нужно не меньше {min_registered}); runtime-карта не создаётся")
        return 1
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
    if not run([sys.executable, str(ROOT / "tools" / "export_map.py"), "--map", args.tag],
               "экспорт runtime.npz", quiet=False):
        return 1

    # Always produce a self-contained model for the raw COLMAP GUI.  Merely
    # writing image_path to project.ini is not robust: an already opened GUI
    # may retain another project path.  Absolute names in the GUI-only model
    # make image viewing independent of the GUI's current options.
    gui = work / "gui"
    gui_model = gui / "sparse" / "0"
    gui_images = gui / "images"
    gui_images.mkdir(parents=True, exist_ok=True)
    gui_model.mkdir(parents=True, exist_ok=True)
    for name in names:
        src, dst = (imdir / name).resolve(), gui_images / name
        try:
            dst.hardlink_to(src)
        except OSError:
            shutil.copy2(src, dst)
    gui_rec = pycolmap.Reconstruction(str(sparse0))
    for image in gui_rec.images.values():
        image.name = str((gui_images / image.name).resolve())
    gui_rec.write_binary(str(gui_model))
    (gui_model / "project.ini").write_text(
        "log_to_stderr=true\n"
        f"database_path={db_path.resolve()}\n"
        f"image_path={gui_images.resolve()}\n",
        encoding="utf-8",
    )
    missing = [image.name for image in gui_rec.images.values() if not Path(image.name).is_file()]
    if missing:
        log(f"ОШИБКА GUI-пакета: отсутствует {len(missing)} изображений")
        return 1
    log(f"--- GUI: {gui_model}, фотографий {len(names)}, абсолютные пути встроены в модель")
    return 0


if __name__ == "__main__":
    sys.exit(main())
