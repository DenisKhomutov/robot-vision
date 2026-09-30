from ..runtime.diagnostics import camera_navigator_diagnostics
from .command_filter import NavigationCommandFilter


class CameraNavigator:
    """Single-camera navigation coordinator.

    The historical class name is retained to avoid changing service imports.
    Camera selection and front/rear failover intentionally do not exist in the
    new runtime.
    """

    def __init__(self, localizer, cfg, map_name=None, route=None):
        self.localizer = localizer
        self.cfg = cfg
        self.pilot = NavigationCommandFilter(cfg)
        self.map = map_name
        self.node_offset = self._node_offset(map_name)
        self.route = route
        self.loading_route = None
        self.route_started = False

    def _node_offset(self, map_name):
        if not map_name:
            return 0
        import json
        from pathlib import Path

        path = Path(self.cfg.MAPS_DIR) / map_name / "shard.json"
        if not path.exists():
            return 0
        return int(json.loads(path.read_text()).get("global_node_start", 0))

    def set_route(self, route):
        spec = getattr(self.cfg, "ROUTES", {}).get(route)
        if spec is None or self.localizer is None or not spec.get("map"):
            return False
        self.route = route
        self.loading_route = None
        self.route_started = False
        self.reset_shard()
        return True

    def clear_route(self):
        self.pause()
        if self.localizer is not None and hasattr(self.localizer, "close"):
            self.localizer.close()
        self.localizer = None
        self.map = None
        self.node_offset = 0
        self.route = None
        self.loading_route = None
        self.route_started = False
        self.pilot = NavigationCommandFilter(self.cfg)

    def set_localizer(self, localizer, map_name, route=None):
        was_paused = self.pilot.paused
        old = self.localizer
        self.localizer = localizer
        self.map = map_name
        self.node_offset = self._node_offset(map_name)
        self.route = route
        self.pilot = NavigationCommandFilter(self.cfg)
        if not was_paused:
            self.pilot.resume()
        if old is not None and hasattr(old, "close"):
            old.close()

    def _dynamic_context(self, result):
        self.map = result.pop("_map_name", self.map)
        self.node_offset = int(result.pop("_node_offset", self.node_offset))
        if result.pop("_map_switched", False):
            was_paused = self.pilot.paused
            self.pilot = NavigationCommandFilter(self.cfg)
            if not was_paused:
                self.pilot.resume()

    def _map_fields(self, cmd):
        cmd["map"] = self.map
        if cmd.get("node") is not None:
            cmd["global_node"] = int(cmd["node"]) + self.node_offset
        if cmd.get("target_node") is not None:
            cmd["global_target_node"] = int(cmd["target_node"]) + self.node_offset
        return cmd

    def pause(self):
        self.pilot.pause()

    def resume(self):
        self.pilot.resume()
        if self.route is not None:
            self.route_started = True

    def reset(self):
        self.pilot.reset()

    def reset_shard(self):
        relocate = getattr(self.localizer, "force_relocate", None)
        if relocate is not None:
            relocate()

    def diagnostics(self):
        return camera_navigator_diagnostics(self)

    def step(self, frame):
        if self.route is None:
            return {
                "move_type": "stop",
                "reason": "route_loading" if self.loading_route else "route_not_selected",
                "route": None,
                "requested_route": self.loading_route,
                "map": None,
                "paused": True,
            }
        if self.pilot.paused:
            cmd = {
                "move_type": "stop",
                "reason": "operator_paused" if self.route_started else "route_loaded",
                "route": self.route,
                "map": self.map,
                "paused": True,
            }
            if not self.route_started:
                cmd["route_loaded"] = True
            return cmd
        if self.localizer is None:
            return {"move_type": "stop", "reason": "map_not_loaded", "route": self.route, "map": self.map}
        if frame is None:
            return {"move_type": "stop", "reason": "camera_frame_missing", "route": self.route, "map": self.map}

        result = self.localizer.locate(frame)
        recovery = bool(result.get("_full_map_recovery"))
        recovery_map = result.get("_recovery_map")
        self._dynamic_context(result)
        cmd = self._map_fields(self.pilot.step(result))
        cmd["route"] = self.route
        if recovery:
            cmd["full_map_recovery"] = True
            cmd["recovery_map"] = recovery_map
        return cmd
