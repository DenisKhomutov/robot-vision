import argparse
import sqlite3
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ExifTags
import torch

ROOT = Path(__file__).resolve().parents[1]
MAX_KPTS = 4096
MAX_OBS_PER_POINT = 12  # сколько ракурсов хранить на 3D-точку
RATIO = 0.85          # тест Лоу по РАССТОЯНИЮ: d1/d2 < RATIO (у SIFT все компоненты >=0,
                      # поэтому косинус высок даже у непохожих — сравнивать надо расстояния)
LOOKAHEAD_NODES = 12  # упреждение в узлах эталона. Нарезка адаптивна (постоянный сдвиг
                      # картинки), поэтому узлы ~равноудалены и это устойчивее метров.
                      # Слишком большое упреждение срезает углы и ведёт сквозь стены.
DEADZONE_DEG = 4.0    # |азимут| меньше -> straight
STANLEY_K = 1.0       # усиление поперечного члена Стэнли (per ед. карты, тюнится)
LOOKAHEAD_MIN = 5     # адаптивное упреждение: не короче этого (иначе рыскание)
LOOKAHEAD_ADAPT = 8.0 # на сколько узлов укорачивать упреждение на ед. бокового смещения
STOP_END_NODES = 3    # ближайший узел в этих последних узлах эталона -> команда stop


class Localizer:
    def __init__(self, map_name, device=None, route_range=None):
        import pycolmap
        work = ROOT / "maps" / map_name
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.rec = pycolmap.Reconstruction(str(work / "sparse" / "0"))
        self.cam = list(self.rec.cameras.values())[0]
        self.sift = cv2.SIFT_create(nfeatures=MAX_KPTS)

        db = sqlite3.connect(str(work / "database.db"))
        ids = {n: i for i, n in db.execute("SELECT image_id, name FROM images")}
        by_name = {im.name: im for im in self.rec.images.values()}

        # эталонная линия: позы кадров по порядку съёмки + куда камера смотрела.
        # route_range=(a,b) — только часть записи: при съёмке туда-обратно эталоном
        # должен быть ОДИН проход, иначе линия возвращается по себе и упреждение
        # разворачивает робота. Обратный проход остаётся донором точек.
        order = sorted(by_name)
        if route_range:
            a, b = route_range
            order = [n for n in order if a <= int("".join(filter(str.isdigit, n)) or 0) < b]
        self.route = np.array([(-by_name[n].cam_from_world().rotation.matrix().T
                                @ by_name[n].cam_from_world().translation) for n in order])
        self.route_fwd = np.array([by_name[n].cam_from_world().rotation.matrix().T @ np.array([0, 0, 1.0])
                                   for n in order])
        seg = np.linalg.norm(np.diff(self.route, axis=0), axis=1)
        self.arc = np.concatenate([[0], np.cumsum(seg)])

        # дескрипторы 3D-точек: берём ВСЕ наблюдения трека, а не одно.
        # Точка видна в среднем в ~10 кадрах, в каждом под своим ракурсом; если хранить
        # только первый, запрос с другого угла не сматчится — это и есть причина промахов.
        cache = {}

        def descs_of(image_id):
            im = self.rec.images[image_id]
            if im.name not in cache:
                r, c, d = db.execute("SELECT rows, cols, data FROM descriptors WHERE image_id=?",
                                     (ids[im.name],)).fetchone()
                a = np.frombuffer(d, np.uint8).reshape(r, c).astype(np.float32)
                cache[im.name] = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
            return cache[im.name]

        desc, owner, xyz = [], [], []
        for pi, (pid, p3) in enumerate(self.rec.points3D.items()):
            xyz.append(p3.xyz)
            for el in p3.track.elements[:MAX_OBS_PER_POINT]:
                arr = descs_of(el.image_id)
                if el.point2D_idx < len(arr):
                    desc.append(arr[el.point2D_idx])
                    owner.append(pi)
        db.close()
        self.mdesc = torch.from_numpy(np.stack(desc)).to(self.dev)
        self.owner = torch.from_numpy(np.array(owner, np.int64)).to(self.dev)
        self.mxyz = np.stack(xyz)
        print(f"[карта] {len(order)} кадров, {len(self.mxyz)} точек, "
              f"{len(desc)} дескрипторов ({len(desc)/max(len(self.mxyz),1):.1f} на точку), {self.dev}")

    @staticmethod
    def focal_from_exif(path, width):
        """Фокус в пикселях из EXIF. Без этого догадка 1.2*сторона может ошибиться
        втрое: снимки со сверхширокого модуля (14мм экв.) дают ~0.39*ширина."""
        try:
            inv = {v: k for k, v in ExifTags.TAGS.items()}
            ex = Image.open(path)._getexif() or {}
            f35 = ex.get(inv.get("FocalLengthIn35mmFilm"))
            if f35:
                return float(f35) / 36.0 * width
        except Exception:
            pass
        return None

    def locate(self, path, chunk=20000, exif_focal=None):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        t0 = time.perf_counter()
        kp, de = self.sift.detectAndCompute(img, None)
        qk = np.array([p.pt for p in kp], np.float32)
        qd = de.astype(np.float32)
        qd /= np.linalg.norm(qd, axis=1, keepdims=True) + 1e-9
        q = torch.from_numpy(qd).to(self.dev)
        t_sift = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        n = len(qk)
        best = torch.full((n,), -1e9, device=self.dev)
        bown = torch.full((n,), -1, dtype=torch.long, device=self.dev)
        second = torch.full((n,), -1e9, device=self.dev)   # лучший среди ДРУГИХ точек
        with torch.inference_mode():
            for i in range(0, len(self.mdesc), chunk):
                sim = q @ self.mdesc[i:i + chunk].T
                own = self.owner[i:i + chunk]
                k = min(8, sim.shape[1])
                tv, ti = sim.topk(k, dim=1)
                for c in range(k):
                    v, o = tv[:, c], own[ti[:, c]]
                    newbest = v > best
                    # прежний лучший уходит во second, только если это ДРУГАЯ точка
                    second = torch.where(newbest & (bown != o) & (bown >= 0),
                                         torch.maximum(second, best), second)
                    second = torch.where(~newbest & (o != bown), torch.maximum(second, v), second)
                    bown = torch.where(newbest, o, bown)
                    best = torch.where(newbest, v, best)
        d1 = torch.sqrt((2 - 2 * best).clamp(min=0))
        d2 = torch.sqrt((2 - 2 * second).clamp(min=0))
        keep = ((d1 / d2.clamp(min=1e-6) < RATIO) & (bown >= 0)).cpu().numpy()
        t_match = (time.perf_counter() - t0) * 1000

        idx = bown.cpu().numpy()
        p2d, p3d = qk[keep], self.mxyz[idx[keep]]
        if len(p2d) < 8:
            return {"ok": False, "reason": f"мало пар: {len(p2d)}", "t_sift": t_sift, "t_match": t_match}

        # запрос может быть снят другой камерой (иное разрешение/пропорции), чем карта:
        # заводим свою камеру с грубым фокусом и разрешаем PnP его уточнить
        import pycolmap
        h_img, w_img = img.shape[:2]
        if (w_img, h_img) == (self.cam.width, self.cam.height):
            cam, ropt = self.cam, pycolmap.AbsolutePoseRefinementOptions()
        else:
            f0 = exif_focal or self.focal_from_exif(path, w_img) or 1.2 * max(w_img, h_img)
            cam = pycolmap.Camera.create_from_model_id(
                2, pycolmap.CameraModelId.SIMPLE_RADIAL, f0, w_img, h_img)
            ropt = pycolmap.AbsolutePoseRefinementOptions()
            ropt.refine_focal_length = True
            ropt.refine_extra_params = True   # сверхширокий даёт заметную бочку

        t0 = time.perf_counter()
        res = pycolmap.estimate_and_refine_absolute_pose(
            p2d, p3d, cam, refinement_options=ropt)
        t_pnp = (time.perf_counter() - t0) * 1000
        if res is None:
            return {"ok": False, "reason": "PnP не сошёлся", "n_pairs": len(p2d),
                    "t_sift": t_sift, "t_match": t_match, "t_pnp": t_pnp}

        R = res["cam_from_world"].rotation.matrix()
        C = -R.T @ res["cam_from_world"].translation
        fwd = R.T @ np.array([0, 0, 1.0])
        out = {"ok": True, "C": C, "fwd": fwd, "inliers": res["num_inliers"], "n_pairs": len(p2d),
               "focal": float(cam.params[0]), "t_sift": t_sift, "t_match": t_match, "t_pnp": t_pnp}
        out.update(self.command(C, fwd))
        return out

    def command(self, C, fwd, lookahead_nodes=LOOKAHEAD_NODES, scale=1.0,
                mode="pursuit", stanley_k=STANLEY_K):
        """Ближайшая точка эталона, смещение вбок, руление (упреждение или Стэнли)."""
        # ближайший узел — С УЧЁТОМ КУРСА: встречный проход проходит рядом, без этого
        # робот цепляется за него и разворачивается
        d = np.linalg.norm(self.route - C, axis=1)
        f = fwd / (np.linalg.norm(fwd) + 1e-9)
        align = self.route_fwd @ f / (np.linalg.norm(self.route_fwd, axis=1) + 1e-9)
        ok = align > 0.3
        k = int(np.argmin(np.where(ok, d, np.inf))) if ok.any() else int(np.argmin(d))
        # знаки: ось для векторных произведений — ВНИЗ (в мире COLMAP +y вниз).
        # С осью «вверх» знак зеркалится: цель слева давала ang>0 -> "right".
        # Соглашение: положительное = ВПРАВО (и азимут, и боковое смещение).
        down = np.array([0, 1.0, 0])
        t = self.route_fwd[k] / (np.linalg.norm(self.route_fwd[k]) + 1e-9)
        v = C - self.route[k]
        e = float(np.dot(np.cross(t, v), down))          # поперечное смещение, + = робот правее
        f_flat = fwd - np.dot(fwd, down) * down

        if mode == "stanley":
            # δ = ψ (курс к касательной у БЛИЖАЙШЕЙ точки) + atan(k·e);
            # e>0 (робот правее линии) -> член <0 -> руль влево, возврат на линию
            j = min(k + lookahead_nodes, len(self.route) - 1)
            t_flat = t - np.dot(t, down) * down
            psi = np.degrees(np.arctan2(np.dot(np.cross(f_flat, t_flat), down),
                                        np.dot(f_flat, t_flat)))
            ang = psi + np.degrees(np.arctan(-stanley_k * e))
        else:
            # pure pursuit, упреждение в ДЛИНЕ ДУГИ (не в узлах — те в поворотах гуще).
            # дальше от линии -> короче упреждение -> резче возврат; на линии -> плавно
            step = getattr(self, "node_step", 1.0)
            want = max(lookahead_nodes - LOOKAHEAD_ADAPT * abs(e), LOOKAHEAD_MIN) * step
            cum = getattr(self, "route_cum", None)
            if cum is not None:
                j = int(np.searchsorted(cum, cum[k] + want))
                j = min(j, len(self.route) - 1)
            else:
                j = min(k + int(lookahead_nodes), len(self.route) - 1)
            to = self.route[j] - C
            to_flat = to - np.dot(to, down) * down
            ang = np.degrees(np.arctan2(np.dot(np.cross(f_flat, to_flat), down),
                                        np.dot(f_flat, to_flat)))
        dz = getattr(self, "deadzone", None) or DEADZONE_DEG   # порог из конфига (через локализатор)
        mt = "straight" if abs(ang) < dz else ("right" if ang > 0 else "left")
        # ДОСТИГЛИ КОНЦА эталона -> стоп (ближайший узел в хвосте маршрута)
        if k >= len(self.route) - 1 - STOP_END_NODES:
            mt, ang = "stop", 0.0
        return {"node": k, "target_node": j, "dist_to_route": float(d[k]),
                "offset": e, "bearing_deg": float(ang), "move_type": mt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photos", nargs="+")
    ap.add_argument("--map", default="map_4f_adaptive_v2")
    ap.add_argument("--device", default=None)
    ap.add_argument("--scale", type=float, default=1.0, help="метров в единице карты")
    args = ap.parse_args()

    loc = Localizer(args.map, args.device)
    print(f"\n{'фото':<22}{'инл.':>6}{'пар':>7}{'узел':>6}{'до линии':>10}{'вбок':>8}"
          f"{'азимут':>9}{'команда':>10}{'мс':>7}")
    for p in args.photos:
        r = loc.locate(Path(p))
        n = Path(p).name[:20]
        if r is None:
            print(f"{n:<22} не читается")
            continue
        ms = r["t_sift"] + r["t_match"] + r.get("t_pnp", 0)
        if not r["ok"]:
            print(f"{n:<22} НЕ НАЙДЕНО: {r['reason']:<40}{ms:>7.0f}")
            continue
        print(f"{n:<22}{r['inliers']:>6}{r['n_pairs']:>7}{r['node']:>6}"
              f"{r['dist_to_route']*args.scale:>9.2f}м{r['offset']*args.scale:>+7.2f}м"
              f"{r['bearing_deg']:>+8.1f}°{r['move_type']:>10}{ms:>7.0f}")


if __name__ == "__main__":
    sys.exit(main())
