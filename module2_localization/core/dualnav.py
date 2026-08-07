from .pilot import Pilot


class DualNav:
    def __init__(self, front_loc, rear_loc, cfg):
        self.front, self.rear, self.cfg = front_loc, rear_loc, cfg
        self.pf, self.pr = Pilot(cfg), Pilot(cfg)
        self.mode = getattr(cfg, "NAV_MODE", "rear")
        self.active = "front"         # какая камера сейчас ведёт (в dual): старт с фронта
        self.front_lost = self.front_good = 0

    def set_mode(self, mode):
        if mode == "dual" and self.front is None:
            return                    # нет передней карты/источника — dual недоступен
        if mode in ("rear", "dual"):
            self.mode = mode
            self.active = "front" if mode == "dual" else "rear"
            self.front_lost = self.front_good = 0

    def _rear(self, rf):
        cmd = self.pr.step(self.rear.locate(rf))
        cmd["cam"] = "rear"
        return cmd

    def step(self, front_frame, rear_frame):
        if self.mode != "dual" or self.front is None or front_frame is None:
            cmd = self._rear(rear_frame)
            cmd["mode"] = "rear"
            return cmd

        cf = self.pf.step(self.front.locate(front_frame))
        front_ok = cf["move_type"] != "lost"
        if front_ok:
            self.front_good += 1
            self.front_lost = 0
        else:
            self.front_lost += 1
            self.front_good = 0
        if self.active == "front" and self.front_lost >= self.cfg.DUAL_LOST_HOLD:
            self.active = "rear"
        elif self.active == "rear" and self.front_good >= self.cfg.DUAL_BACK_HOLD:
            self.active = "front"

        if self.active == "front" and front_ok:
            cmd = cf
            cmd["cam"] = "front"
        else:                          # ленивый резерв: заднюю гоняем только когда ведёт она
            cmd = self._rear(rear_frame)
        cmd["mode"] = "dual"
        return cmd