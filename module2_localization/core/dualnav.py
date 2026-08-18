from .pilot import Pilot


def _route_id(map_name):
    if not map_name:
        return None
    if "route12" in map_name:
        return "route12"
    if "route3" in map_name:
        return "route3"
    return None


class DualNav:
    def __init__(self, front_loc, rear_loc, cfg, front_map=None, rear_map=None):
        self.front, self.rear, self.cfg = front_loc, rear_loc, cfg
        self.pf, self.pr = Pilot(cfg), Pilot(cfg)
        self.mode = getattr(cfg, "NAV_MODE", "rear")
        self.active = "front"         # какая камера сейчас ведёт (в dual): старт с фронта
        self.front_lost = self.front_good = 0
        self.front_map = front_map or getattr(cfg, "FRONT_MAP", None)
        self.rear_map = rear_map or getattr(cfg, "REAR_MAP", None)
        self.front_node_offset = self._node_offset(self.front_map)
        self.rear_node_offset = self._node_offset(self.rear_map)
        self.route = getattr(cfg, "DEFAULT_ROUTE", "route12")

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

    def set_localizer(self, camera, localizer, map_name):
        if camera == "front":
            was_paused = self.pf.paused
            old = self.front
            self.front = localizer
            self.front_map = map_name
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

    def set_mode(self, mode):
        if mode == "dual" and self.front is None:
            return                    # нет передней карты/источника — dual недоступен
        if mode in ("rear", "dual"):
            self.mode = mode
            self.active = "front" if mode == "dual" else "rear"
            self.front_lost = self.front_good = 0

    def _rear(self, rf):
        result = self.rear.locate(rf)
        map_name, offset = self._dynamic_context("rear", result)
        cmd = self.pr.step(result)
        return self._map_fields(cmd, "rear", map_name, offset)

    def _maps_compatible(self):
        front_route, rear_route = _route_id(self.front_map), _route_id(self.rear_map)
        return front_route is None or rear_route is None or front_route == rear_route

    def step(self, front_frame, rear_frame):
        if self.mode != "dual" or self.front is None or front_frame is None:
            cmd = self._rear(rear_frame)
            cmd["mode"] = "rear"
            cmd["route"] = self.route
            return cmd

        front_result = self.front.locate(front_frame)
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
        elif not self._maps_compatible():
            # Нельзя локализовать route12-кадр по route3-карте: при потере front
            # безопасно останавливаемся, пока оператор не выберет совместимую rear-карту.
            cmd = {"move_type": "stop", "cam": "front", "map": self.front_map,
                   "reason": "front/rear maps belong to different routes", "map_mismatch": True}
        else:                          # ленивый резерв: заднюю гоняем только когда ведёт она
            cmd = self._rear(rear_frame)
        cmd["mode"] = "dual"
        cmd["route"] = self.route
        return cmd
