from .pilot import Pilot


class DualNav:
    def __init__(self, front_loc, rear_loc, cfg, front_map=None, rear_map=None,
                 route=None, front_route=None, rear_route=None):
        self.front, self.rear, self.cfg = front_loc, rear_loc, cfg
        self.pf, self.pr = Pilot(cfg), Pilot(cfg)
        self.mode = getattr(cfg, "NAV_MODE", "rear")
        self.active = "front"         # какая камера сейчас ведёт (в dual): старт с фронта
        self.front_lost = self.front_good = 0
        self.front_map = front_map
        self.rear_map = rear_map
        self.front_node_offset = self._node_offset(self.front_map)
        self.rear_node_offset = self._node_offset(self.rear_map)
        self.route = route or getattr(cfg, "DEFAULT_ROUTE", "1")
        # Какому маршруту принадлежит карта на каждой камере — используется для
        # _maps_compatible(), чтобы не проверять по имени файла карты (имена
        # произвольные, напр. office-карты "map_2cam_front"/"map_2cam_rear").
        self.front_route = front_route if front_loc is not None else None
        self.rear_route = rear_route if rear_loc is not None else None

    def set_route(self, route):
        routes = getattr(self.cfg, "ROUTES", {})
        if route not in routes:
            return False
        camera = routes[route]["camera"]
        if camera == "front" and self.front is None:
            return False
        self.route = route
        self.mode = "dual" if camera == "front" else "rear"
        self.active = camera
        self.front_lost = self.front_good = 0
        return True

    def _node_offset(self, map_name):
        if not map_name:
            return 0
        import json
        from pathlib import Path
        path = Path(self.cfg.MAPS_DIR) / map_name / "shard.json"
        if not path.exists():
            return 0
        return int(json.loads(path.read_text()).get("global_node_start", 0))

    def set_localizer(self, camera, localizer, map_name, route=None):
        if camera == "front":
            was_paused = self.pf.paused
            old = self.front
            self.front = localizer
            self.front_map = map_name
            self.front_route = route
            self.front_node_offset = self._node_offset(map_name)
            self.pf = Pilot(self.cfg)
            if not was_paused:
                self.pf.resume()
            self.active = "front"
        elif camera == "rear":
            was_paused = self.pr.paused
            old = self.rear
            self.rear = localizer
            self.rear_map = map_name
            self.rear_route = route
            self.rear_node_offset = self._node_offset(map_name)
            self.pr = Pilot(self.cfg)
            if not was_paused:
                self.pr.resume()
        else:
            raise ValueError(f"неизвестная камера {camera}")
        if old is not None and hasattr(old, "close"):
            old.close()
        self.front_lost = self.front_good = 0

    @staticmethod
    def _map_fields(cmd, camera, map_name, offset):
        cmd["cam"] = camera
        cmd["map"] = map_name
        if cmd.get("node") is not None:
            cmd["global_node"] = int(cmd["node"]) + offset
        if cmd.get("target_node") is not None:
            cmd["global_target_node"] = int(cmd["target_node"]) + offset
        return cmd

    def _dynamic_context(self, camera, result):
        if camera == "front":
            self.front_map = result.pop("_map_name", self.front_map)
            self.front_node_offset = int(result.pop("_node_offset", self.front_node_offset))
            if result.pop("_map_switched", False):
                was_paused = self.pf.paused
                self.pf = Pilot(self.cfg)
                if not was_paused:
                    self.pf.resume()
            return self.front_map, self.front_node_offset
        self.rear_map = result.pop("_map_name", self.rear_map)
        self.rear_node_offset = int(result.pop("_node_offset", self.rear_node_offset))
        if result.pop("_map_switched", False):
            was_paused = self.pr.paused
            self.pr = Pilot(self.cfg)
            if not was_paused:
                self.pr.resume()
        return self.rear_map, self.rear_node_offset

    def pause(self):
        self.pf.pause()
        self.pr.pause()

    def resume(self):
        self.pf.resume()
        self.pr.resume()
        self.reset_shard()

    def reset_shard(self):
        """Заставить обе камеры заново определиться по полной карте: оператор
        физически передвинул робота (на соседний шард/после сбоя), локальный
        node текущего шарда больше ничего не значит. Пока идёт релокализация,
        команды безопасны — locate() отдаёт lost, реального движения нет."""
        for loc in (self.front, self.rear):
            relocate = getattr(loc, "force_relocate", None)
            if relocate is not None:
                relocate()

    def has_camera(self, camera):
        return (self.front if camera == "front" else self.rear) is not None

    def set_mode(self, mode):
        if mode not in ("rear", "dual", "front"):
            return False
        if mode in ("dual", "front") and self.front is None:
            return False               # нет передней карты/источника
        if mode == "rear" and self.rear is None:
            return False               # нет задней карты/источника
        self.mode = mode
        self.active = "rear" if mode == "rear" else "front"
        self.front_lost = self.front_good = 0
        return True

    def _front_only(self, ff):
        if self.front is None:
            return {"move_type": "stop", "reason": "нет карты передней камеры", "cam": "front"}
        result = self.front.locate(ff)
        recovery = bool(result.get("_full_map_recovery"))
        recovery_map = result.get("_recovery_map")
        map_name, offset = self._dynamic_context("front", result)
        cmd = self.pf.step(result)
        cmd = self._map_fields(cmd, "front", map_name, offset)
        if recovery:
            cmd["full_map_recovery"] = True
            cmd["recovery_map"] = recovery_map
        return cmd

    def _rear(self, rf):
        if self.rear is None:
            return {"move_type": "stop", "reason": "нет карты задней камеры", "cam": "rear"}
        result = self.rear.locate(rf)
        recovery = bool(result.get("_full_map_recovery"))
        recovery_map = result.get("_recovery_map")
        map_name, offset = self._dynamic_context("rear", result)
        cmd = self.pr.step(result)
        cmd = self._map_fields(cmd, "rear", map_name, offset)
        if recovery:
            cmd["full_map_recovery"] = True
            cmd["recovery_map"] = recovery_map
        return cmd

    def _maps_compatible(self):
        return self.front_route is None or self.rear_route is None or self.front_route == self.rear_route

    def step(self, front_frame, rear_frame):
        if self.mode == "front":
            cmd = self._front_only(front_frame) if front_frame is not None else \
                {"move_type": "stop", "reason": "нет кадра передней камеры", "cam": "front"}
            cmd["mode"] = "front"
            cmd["route"] = self.route
            return cmd

        if self.mode != "dual" or self.front is None or front_frame is None:
            cmd = self._rear(rear_frame)
            cmd["mode"] = "rear"
            cmd["route"] = self.route
            return cmd

        front_result = self.front.locate(front_frame)
        front_recovery = bool(front_result.get("_full_map_recovery"))
        front_recovery_map = front_result.get("_recovery_map")
        front_map, front_offset = self._dynamic_context("front", front_result)
        cf = self.pf.step(front_result)
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
            cmd = self._map_fields(cf, "front", front_map, front_offset)
            if front_recovery:
                cmd["full_map_recovery"] = True
                cmd["recovery_map"] = front_recovery_map
        elif not self._maps_compatible():
            # Нельзя локализовать кадр одного маршрута по карте другого: при
            # потере front безопасно останавливаемся, пока оператор не выберет
            # совместимую rear-карту (тот же маршрут).
            cmd = {"move_type": "stop", "cam": "front", "map": self.front_map,
                   "reason": "front/rear maps belong to different routes", "map_mismatch": True}
        else:                          # ленивый резерв: заднюю гоняем только когда ведёт она
            cmd = self._rear(rear_frame)
        cmd["mode"] = "dual"
        cmd["route"] = self.route
        return cmd
