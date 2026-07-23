import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pycolmap
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from route import Localizer  # noqa: E402  (focal_from_exif + command/route)

RATIO = 0.9
LOOKAHEAD_NODES = 12
DEADZONE_DEG = 4.0


class AlikedLocalizer:
    def __init__(self, map_name="map_office_ref", device=None, kpts=2048, route_range=None,
                 det_threshold=0.2, nms_radius=2, max_error=12.0, steer="pursuit", route_cam=None):
        from lightglue import ALIKED
        self.max_error = max_error
        self.steer = steer  # "pursuit" (упреждение) | "stanley"
        self.route_cam = route_cam  # для риг-карт: маршрут только по кадрам этой камеры (напр. "_c2")
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        work = ROOT / "maps" / map_name
        self.rec = pycolmap.Reconstruction(str(work / "sparse" / "0"))
        self.cam = list(self.rec.cameras.values())[0]
        # плотность ЗАПРОСА независима от карты: лишние точки, что не лягут, отбрасываются,
        # а попавшие в сильные точки карты добавляют инлайеров. Карту это не трогает.
        self.ext = ALIKED(max_num_keypoints=kpts, detection_threshold=det_threshold,
                          nms_radius=nms_radius).eval().to(self.dev)

        sf = work / "scale.json"
        self.scale = json.loads(sf.read_text())["scale_m_per_unit"] if sf.exists() else None

        bank = np.load(work / "aliked_bank.npz")
        self.mdesc = torch.from_numpy(bank["desc"].astype(np.float32)).to(self.dev)
        self.owner = torch.from_numpy(bank["owner"].astype(np.int64)).to(self.dev)
        self.mxyz = bank["xyz"]

        by_name = {im.name: im for im in self.rec.images.values()}
        order = sorted(by_name)
        if self.route_cam:  # риг-карта: маршрут по ОДНОЙ камере, иначе зигзаг между 3 путями
            order = [n for n in order if self.route_cam in n]
        if route_range:
            a, b = route_range
            order = [n for n in order if a <= int("".join(filter(str.isdigit, n)) or 0) < b]
        # выкидываем кадры-ВЫБРОСЫ (плохая регистрация: улетели далеко от обоих соседей).
        # Иначе упреждающая цель может попасть на выброс -> команда «в бесконечность».
        pos = np.array([(-by_name[n].cam_from_world().rotation.matrix().T
                         @ by_name[n].cam_from_world().translation) for n in order])
        if len(pos) > 5:
            seg = np.linalg.norm(np.diff(pos, axis=0), axis=1)
            thr = 10 * np.median(seg)
            keep = [True] * len(order)
            for k in range(1, len(order) - 1):
                if seg[k - 1] > thr and seg[k] > thr:
                    keep[k] = False
            dropped = len(order) - sum(keep)
            if dropped:
                print(f"[карта] выкинуто выбросов маршрута: {dropped}")
            order = [n for n, kp in zip(order, keep) if kp]
        self.route = np.array([(-by_name[n].cam_from_world().rotation.matrix().T
                                @ by_name[n].cam_from_world().translation) for n in order])
        self.route_fwd = np.array([by_name[n].cam_from_world().rotation.matrix().T
                                   @ np.array([0, 0, 1.0]) for n in order])
        print(f"[карта] {len(order)} кадров, {len(self.mxyz)} точек, "
              f"{len(self.owner)} дескрипторов, {self.dev}")

    def locate(self, path, chunk=100000, exif_focal=None):
        from lightglue.utils import load_image
        t0 = time.perf_counter()
        # принимаем и путь (офлайн-инструменты), и BGR-кадр numpy (онлайн: WebRTC)
        if isinstance(path, np.ndarray):
            rgb = cv2.cvtColor(path, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            im = torch.from_numpy(rgb.transpose(2, 0, 1)).to(self.dev)
        else:
            im = load_image(str(path)).to(self.dev)
        with torch.inference_mode():
            f = self.ext.extract(im)
        qk = f["keypoints"][0].cpu().numpy()
        q = f["descriptors"][0]
        t_ext = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        n = len(qk)
        best = torch.full((n,), -1e9, device=self.dev)
        bown = torch.full((n,), -1, dtype=torch.long, device=self.dev)
        second = torch.full((n,), -1e9, device=self.dev)
        bxyz = np.zeros((n, 3))
        with torch.inference_mode():
            for i in range(0, len(self.owner), chunk):
                sim = q @ self.mdesc[i:i + chunk].T
                own = self.owner[i:i + chunk]
                k = min(8, sim.shape[1])
                tv, ti = sim.topk(k, dim=1)
                for c in range(k):
                    v, o = tv[:, c], own[ti[:, c]]
                    nb = v > best
                    second = torch.where(nb & (bown != o) & (bown >= 0),
                                         torch.maximum(second, best), second)
                    second = torch.where(~nb & (o != bown), torch.maximum(second, v), second)
                    u = nb.cpu().numpy()
                    if u.any():
                        bxyz[u] = self.mxyz[o.cpu().numpy()[u]]
                    bown = torch.where(nb, o, bown)
                    best = torch.where(nb, v, best)
        d1 = torch.sqrt((2 - 2 * best).clamp(min=0))
        d2 = torch.sqrt((2 - 2 * second).clamp(min=0))
        keep = ((d1 / d2.clamp(min=1e-6) < RATIO) & (bown >= 0)).cpu().numpy()
        t_match = (time.perf_counter() - t0) * 1000

        p2d, p3d = qk[keep], bxyz[keep]
        if len(p2d) < 8:
            return {"ok": False, "reason": f"мало пар: {len(p2d)}",
                    "t_ext": t_ext, "t_match": t_match}

        h_img, w_img = im.shape[-2], im.shape[-1]
        if (w_img, h_img) == (self.cam.width, self.cam.height):
            cam, ropt = self.cam, pycolmap.AbsolutePoseRefinementOptions()
        else:
            ef = None if isinstance(path, np.ndarray) else Localizer.focal_from_exif(path, w_img)
            f0 = exif_focal or ef or 1.2 * max(w_img, h_img)
            cam = pycolmap.Camera.create_from_model_id(
                2, pycolmap.CameraModelId.SIMPLE_RADIAL, f0, w_img, h_img)
            ropt = pycolmap.AbsolutePoseRefinementOptions()
            ropt.refine_focal_length = True
            ropt.refine_extra_params = True

        eopt = pycolmap.AbsolutePoseEstimationOptions()
        eopt.ransac.max_error = self.max_error  # порог репроекции инлайера, px (дефолт 12)
        t0 = time.perf_counter()
        res = pycolmap.estimate_and_refine_absolute_pose(
            p2d, p3d, cam, estimation_options=eopt, refinement_options=ropt)
        t_pnp = (time.perf_counter() - t0) * 1000
        if res is None:
            return {"ok": False, "reason": "PnP не сошёлся", "n_pairs": len(p2d),
                    "t_ext": t_ext, "t_match": t_match, "t_pnp": t_pnp}
        R = res["cam_from_world"].rotation.matrix()
        C = -R.T @ res["cam_from_world"].translation
        fwd = R.T @ np.array([0, 0, 1.0])
        out = {"ok": True, "C": C, "fwd": fwd, "inliers": res["num_inliers"],
               "n_pairs": len(p2d), "t_ext": t_ext, "t_match": t_match, "t_pnp": t_pnp}
        out.update(Localizer.command(self, C, fwd, mode=self.steer))
        if self.scale:
            out["dist_to_route_m"] = out["dist_to_route"] * self.scale
            out["offset_m"] = out["offset"] * self.scale
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photos", nargs="+")
    ap.add_argument("--map", default="map_office_ref")
    ap.add_argument("--route-range", default=None, help="эталон только по кадрам a:b (напр. 0:525)")
    args = ap.parse_args()
    rr = tuple(int(x) for x in args.route_range.split(':')) if args.route_range else None
    loc = AlikedLocalizer(args.map, route_range=rr)
    print(f"\n{'фото':<20}{'инл.':>6}{'пар':>7}{'узел':>6}{'до линии':>10}{'азимут':>9}{'команда':>10}{'мс':>7}")
    for p in args.photos:
        r = loc.locate(Path(p))
        n = Path(p).name[:18]
        if r is None:
            print(f"{n:<20} не читается")
            continue
        ms = r["t_ext"] + r["t_match"] + r.get("t_pnp", 0)
        if not r["ok"]:
            print(f"{n:<20} НЕ НАЙДЕНО: {r['reason']:<40}{ms:>7.0f}")
            continue
        d = r.get("dist_to_route_m", r["dist_to_route"])
        u = "м" if "dist_to_route_m" in r else "е"
        print(f"{n:<20}{r['inliers']:>6}{r['n_pairs']:>7}{r['node']:>6}"
              f"{d:>9.2f}{u}{r['bearing_deg']:>+8.1f}°{r['move_type']:>10}{ms:>7.0f}")


if __name__ == "__main__":
    sys.exit(main())
