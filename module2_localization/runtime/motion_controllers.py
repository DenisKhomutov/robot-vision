class DirectionController:
    def __init__(self, cfg, enabled=False):
        self.zones = getattr(cfg, "BACKWARD_ZONES", {})
        self.maps = set(getattr(cfg, "BACKWARD_MAPS", ()))
        self.steering_outlier_guards = getattr(cfg, "STEERING_OUTLIER_GUARDS", {})
        self.deadzone = getattr(cfg, "DEADZONE_DEG", 4.0)
        self.enabled = bool(enabled)

    def set_enabled(self, enabled):
        self.enabled = bool(enabled)

    def process(self, cmd):
        cmd["direction_enabled"] = self.enabled
        if not self.enabled:
            return
        ranges = self.zones.get(cmd.get("map"))
        node = cmd.get("global_node", cmd.get("node"))
        backward = cmd.get("map") in self.maps or bool(
            ranges and node is not None and any(a <= node <= b for a, b in ranges)
        )
        cmd["direction"] = "backward" if backward else "forward"
        if backward and cmd.get("deg") is not None and cmd.get("move_type") not in ("stop", "lost"):
            rear_axis_error = ((float(cmd["deg"]) - 180.0 + 180.0) % 360.0) - 180.0
            deg = rear_axis_error
            cmd["deg"] = deg
            cmd["move_type"] = "straight" if abs(deg) < self.deadzone else ("right" if deg > 0 else "left")
        guard = self.steering_outlier_guards.get(cmd.get("map"))
        if guard and node is not None and cmd.get("deg") is not None:
            (start, stop), max_deg = guard
            if start <= node <= stop and abs(float(cmd["deg"])) > float(max_deg):
                cmd["deg"] = float(max_deg) if float(cmd["deg"]) > 0 else -float(max_deg)
                cmd["move_type"] = "right" if cmd["deg"] > 0 else "left"


class RouteProfileController:
    def __init__(self, cfg, terminal_maneuvers=None):
        if terminal_maneuvers is None:
            terminal_maneuvers = getattr(cfg, "ROUTE_21_TERMINAL_MANEUVERS_DEFAULT", True)
        self.cfg = cfg
        self.terminal_maneuvers = bool(terminal_maneuvers)
        self.completed = False

    def set_terminal_maneuvers(self, enabled):
        self.terminal_maneuvers = bool(enabled)
        self.reset()

    def reset(self):
        self.completed = False

    def process(self, cmd):
        cmd["terminal_maneuvers"] = self.terminal_maneuvers
        if self.terminal_maneuvers or cmd.get("route") != "2-1":
            return
        node = cmd.get("global_node")
        if node is None:
            if self.completed:
                cmd["move_type"] = "stop"
                cmd["deg"] = 0.0
                cmd["reason"] = "route_complete"
            return
        min_node = int(getattr(self.cfg, "ROUTE_21_NO_MANEUVERS_MIN_NODE", 27))
        stop_node = int(getattr(self.cfg, "ROUTE_21_NO_MANEUVERS_STOP_NODE", 1595))
        if int(node) < min_node:
            cmd["move_type"] = "stop"
            cmd["deg"] = 0.0
            cmd["reason"] = "outside_route_segment"
            return
        if int(node) >= stop_node:
            self.completed = True
        if self.completed:
            cmd["move_type"] = "stop"
            cmd["deg"] = 0.0
            cmd["reason"] = "route_complete"
