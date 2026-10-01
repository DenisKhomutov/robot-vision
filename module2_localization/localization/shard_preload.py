import logging
import time

LOGGER = logging.getLogger("sharded-localizer")


class ShardPreloadMixin:
    def _start_preload(self, target: int) -> None:
        if target < self.min_shard_index or target > self.max_shard_index or target == self.index:
            self._event(
                "shard_preload_skipped",
                current_index=self.index,
                target_index=target,
                reason="outside_allowed_range_or_current",
            )
            return
        if self._pending_index == target:
            return
        if self._future and not self._future.done():
            return
        map_name = str(self.shards[target]["map"])
        self._pending_index = target
        self._pending = None
        self._pending_good = 0
        self._preload_started = time.monotonic()
        if target in self._loaded:
            self._pending = self._loaded[target]
            self._future = None
            LOGGER.info("шард %s уже находится в памяти", map_name)
            self._event(
                "shard_preload_ready", map=map_name, target_index=target, source="already_loaded", elapsed_s=0.0
            )
            return
        LOGGER.info("начата фоновая загрузка %s", map_name)
        self._event(
            "shard_preload_started",
            map=map_name,
            current_map=self.current_map,
            current_index=self.index,
            target_index=target,
            last_global_node=self._last_global_node,
        )
        self._future = self._executor.submit(self.factory, map_name, self.back_facing)

    def _evict_other(self, keep_index: int) -> None:
        for i in list(self._loaded):
            if i == keep_index:
                continue
            localizer = self._loaded.pop(i)
            if hasattr(localizer, "close"):
                localizer.close()

    def _clear_pending(self) -> None:
        self._pending = None
        self._pending_index = None
        self._future = None
        self._pending_good = 0
        self._preload_started = None

    def _collect_preload(self) -> None:
        if self._future and self._future.done() and self._pending is None:
            try:
                self._pending = self._future.result()
                self._loaded[int(self._pending_index)] = self._pending
                elapsed = 0.0 if self._preload_started is None else time.monotonic() - self._preload_started
                LOGGER.info("предзагружен шард %s за %.2fс", self.shards[int(self._pending_index)]["map"], elapsed)
                self._event(
                    "shard_preload_ready",
                    map=self.shards[int(self._pending_index)]["map"],
                    target_index=self._pending_index,
                    source="background",
                    elapsed_s=elapsed,
                )
            except Exception as exc:
                LOGGER.error("не удалось предзагрузить шард: %s", exc)
                self._event("shard_preload_failed", target_index=self._pending_index, error=repr(exc))
                self._clear_pending()

    def _schedule(self, global_node: int) -> None:
        item = self.shards[self.index]
        core_stop = int(item["core_global_stop_exclusive"])
        self._last_global_node = global_node

        preload_start = int(item.get("preload_global_start", core_stop - self.preload_nodes))
        self._event(
            "shard_schedule_check",
            current_map=self.current_map,
            global_node=global_node,
            preload_start=preload_start,
            core_stop_exclusive=core_stop,
            pending_index=self._pending_index,
        )
        if global_node >= preload_start:
            self._start_preload(self.index + 1)
