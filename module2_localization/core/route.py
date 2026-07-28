import numpy as np
from PIL import Image, ExifTags

LOOKAHEAD_NODES = 12  # упреждение в узлах эталона (нарезка адаптивна -> узлы ~равноудалены)
DEADZONE_DEG = 4.0    # |азимут| меньше -> straight
STANLEY_K = 1.0       # усиление поперечного члена Стэнли (per ед. карты, тюнится)
LOOKAHEAD_MIN = 5     # адаптивное упреждение: не короче этого (иначе рыскание)
LOOKAHEAD_ADAPT = 8.0 # на сколько узлов укорачивать упреждение на ед. бокового смещения
STOP_END_NODES = 3    # ближайший узел в этих последних узлах эталона -> команда stop


class Localizer:
    """Руление по эталону. AlikedLocalizer наследует command() и focal_from_exif();
    маршрут (self.route, route_fwd, route_cum, node_step) он строит сам."""

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

    def command(self, C, fwd, lookahead_nodes=LOOKAHEAD_NODES, mode="pursuit", stanley_k=STANLEY_K):
        """Ближайшая точка эталона, смещение вбок, руление (упреждение или Стэнли)."""
        # ближайший узел — С УЧЁТОМ КУРСА: встречный проход проходит рядом, без этого
        # робот цепляется за него и разворачивается
        d = np.linalg.norm(self.route - C, axis=1)
        f = fwd / (np.linalg.norm(fwd) + 1e-9)
        align = self.route_fwd @ f / (np.linalg.norm(self.route_fwd, axis=1) + 1e-9)
        ok = align > 0.3
        k = int(np.argmin(np.where(ok, d, np.inf))) if ok.any() else int(np.argmin(d))
        # знаки: ось для векторных произведений — ВНИЗ (в мире COLMAP +y вниз).
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
                j = min(int(np.searchsorted(cum, cum[k] + want)), len(self.route) - 1)
            else:
                j = min(k + int(lookahead_nodes), len(self.route) - 1)
            to = self.route[j] - C
            to_flat = to - np.dot(to, down) * down
            ang = np.degrees(np.arctan2(np.dot(np.cross(f_flat, to_flat), down),
                                        np.dot(f_flat, to_flat)))
        dz = getattr(self, "deadzone", None) or DEADZONE_DEG   # порог из конфига (через локализатор)
        mt = "straight" if abs(ang) < dz else ("right" if ang > 0 else "left")
        # ДОСТИГЛИ КОНЦА эталона -> стоп (ближайший узел в хвосте маршрута)
        if k >= len(self.route) - 1 - getattr(self, "stop_end_nodes", STOP_END_NODES):
            mt, ang = "stop", 0.0
        return {"node": k, "target_node": j, "dist_to_route": float(d[k]),
                "offset": e, "bearing_deg": float(ang), "move_type": mt}
