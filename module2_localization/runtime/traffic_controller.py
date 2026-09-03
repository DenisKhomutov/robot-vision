import asyncio

import cv2


class TrafficBranch:
    def __init__(self, cfg):
        import module1_traffic_light as traffic_light
        from module1_traffic_light import config as traffic_config

        traffic_config.DET_CONF = getattr(cfg, "TRAFFIC_DET_CONF", traffic_config.DET_CONF)
        traffic_light.init_models()
        from module1_traffic_light.services.traffic_light_service import analyze

        self._analyze = analyze
        self.zones = getattr(cfg, "TRAFFIC_ZONES", {})
        self.last_map, self.last_node = None, None
        self.state = "WAIT_RED"
        self.completed = False

    def reset(self):
        self.state = "WAIT_RED"
        self.completed = False
        self.last_map = None
        self.last_node = None

    @staticmethod
    def _in(zone, node):
        return zone is not None and node is not None and zone[0] <= node <= zone[1]

    def in_zone(self, map_name, node):
        if map_name != self.last_map:
            self.last_node = None
            self.last_map = map_name
        if node is not None:
            self.last_node = node
        node = node if node is not None else self.last_node
        zone = self.zones.get(map_name)
        return self._in(zone, node)

    @property
    def go(self):
        return self.completed

    def update(self, in_zone, frame_bgr):
        if self.completed:
            return None
        if not in_zone:
            self.state = "WAIT_RED"
            return None
        from PIL import Image

        result = self._analyze(Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)))
        signal = (result.get("signal") or "").lower()
        is_red = "red" in signal or "крас" in signal or "stop" in signal
        is_green = "green" in signal or "зел" in signal or "go" in signal
        if self.state == "WAIT_RED" and is_red:
            self.state = "WAIT_GREEN"
        elif self.state == "WAIT_GREEN" and is_green:
            self.state = "GO"
            self.completed = True
        return result


class TrafficController:
    def __init__(self, cfg):
        self.cfg = cfg
        self.enabled = False
        self.loading = False
        self.branch = None

    async def preload(self):
        if self.branch is not None:
            return
        self.loading = True
        try:
            self.branch = await asyncio.to_thread(TrafficBranch, self.cfg)
        finally:
            self.loading = False

    async def set_enabled(self, enabled):
        if not enabled:
            self.enabled = False
            if self.branch:
                self.branch.reset()
            return
        if self.branch is None:
            await self.preload()
        self.branch.reset()
        self.enabled = True

    def reset(self):
        if self.branch:
            self.branch.reset()

    @property
    def state(self):
        return None if self.branch is None else self.branch.state

    def process(self, cmd, frame):
        cmd["traffic_enabled"] = self.enabled
        cmd["traffic_loading"] = self.loading
        if not self.enabled or self.branch is None:
            return None
        in_zone = self.branch.in_zone(cmd.get("map"), cmd.get("global_node", cmd.get("node")))
        result = self.branch.update(in_zone, frame)
        cmd["traffic_in_zone"] = in_zone
        cmd["traffic_state"] = self.branch.state
        if in_zone and not self.branch.go and cmd.get("move_type") != "lost":
            cmd["move_type"], cmd["deg"] = "stop", 0.0
            cmd["reason"] = "traffic_light_stop"
        return result
