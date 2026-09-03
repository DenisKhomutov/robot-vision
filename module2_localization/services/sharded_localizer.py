from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Protocol

from ..runtime.diagnostics import emit_event, localization_result_summary
from ..runtime.shard_manifest import (
    load_shard_chain,
    recovery_map_name,
    shard_index_for_global_node,
    validate_runtime_bundle,
)
from ..runtime.shard_preload import ShardPreloadMixin
from ..runtime.shard_recovery import ShardRecoveryMixin

LOGGER = logging.getLogger("sharded-localizer")


class LocalizationBackend(Protocol):
    def locate(self, frame: object) -> dict: ...
    def extract_query(self, frame: object) -> object: ...
    def locate_features(self, query: object) -> dict: ...


class ShardedLocalizer(ShardPreloadMixin, ShardRecoveryMixin):
    def __init__(
        self,
        maps_dir: Path,
        start_map: str,
        back_facing: bool,
        factory: Callable[[str, bool], LocalizationBackend],
        preload_nodes: int = 25,
        confirm_fixes: int = 2,
        min_inliers: int = 20,
        preload_all: bool = False,
        full_recovery: bool = False,
        recovery_min_inliers: int = 20,
        min_shard_index: int | None = None,
        max_shard_index: int | None = None,
        event_sink: Callable[..., None] | None = None,
        switch_policies: dict[str, dict] | None = None,
    ) -> None:
        self.maps_dir = maps_dir
        self.back_facing = back_facing
        self.factory = factory
        self.preload_nodes = preload_nodes
        self.confirm_fixes = confirm_fixes
        self.min_inliers = min_inliers
        self.recovery_min_inliers = recovery_min_inliers
        self.event_sink = event_sink
        self.switch_policies = switch_policies or {}
        self.shards = load_shard_chain(self.maps_dir, start_map)
        self.min_shard_index = 0 if min_shard_index is None else int(min_shard_index)
        self.max_shard_index = len(self.shards) - 1 if max_shard_index is None else int(max_shard_index)
        if not 0 <= self.min_shard_index <= self.max_shard_index < len(self.shards):
            raise ValueError("недопустимые границы цепочки шардов")
        self.index = next(i for i, item in enumerate(self.shards) if item["map"] == start_map)
        if not self.min_shard_index <= self.index <= self.max_shard_index:
            raise ValueError("стартовый шард находится вне разрешённого диапазона")
        self.current: LocalizationBackend | None = None
        self._loaded: dict[int, LocalizationBackend] = {}
        self.current_map = start_map
        self.node_offset = int(self.shards[self.index]["global_node_start"])
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="map-preload")
        self._future: Future | None = None
        self._pending_index: int | None = None
        self._pending = None
        self._pending_good = 0
        self._last_global_node: int | None = None
        self._preload_started: float | None = None
        self._force_recovery = False
        self._recovery = None
        self._lost_recovery_enabled = bool(full_recovery)
        self._recovery_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="map-recovery")
        self._recovery_future: Future | None = None
        self.recovery_map = recovery_map_name(self.shards)
        self._recovery_calls = 0
        self._recovery_fixes = 0
        self._switch_progress_node: int | None = None
        validate_runtime_bundle(self.maps_dir, self.recovery_map)
        self._event(
            "shard_chain_initialized",
            start_map=start_map,
            shard_count=len(self.shards),
            min_shard_index=self.min_shard_index,
            max_shard_index=self.max_shard_index,
            preload_nodes=self.preload_nodes,
            confirm_fixes=self.confirm_fixes,
            min_inliers=self.min_inliers,
            recovery_map=self.recovery_map,
        )
        started = time.monotonic()
        self._recovery = factory(self.recovery_map, back_facing)
        LOGGER.info("полная recovery-карта %s загружена за %.2fс", self.recovery_map, time.monotonic() - started)

        if preload_all:
            started = time.monotonic()
            for i, item in enumerate(self.shards):
                if not self.min_shard_index <= i <= self.max_shard_index:
                    continue
                if i not in self._loaded:
                    LOGGER.info("загрузка шарда в память %s", item["map"])
                    self._loaded[i] = factory(str(item["map"]), back_facing)
            if self.current is None:
                self.current = self._loaded[self.index]
            LOGGER.info("вся цепочка из %d шардов загружена за %.2fс", len(self.shards), time.monotonic() - started)

    def _event(self, event: str, **fields) -> None:
        emit_event(self.event_sink, "sharded_localizer", event, LOGGER, **fields)

    @staticmethod
    def _result_summary(result: dict | None) -> dict:
        return localization_result_summary(result)

    def _update_switch_progress(self, global_node: int) -> None:
        policy = self.switch_policies.get(self.current_map)
        if not policy:
            self._switch_progress_node = global_node
            return
        previous = self._switch_progress_node
        max_step = int(policy.get("max_progress_step", self.preload_nodes))
        accepted = previous is None or global_node <= previous or global_node - previous <= max_step
        if accepted:
            self._switch_progress_node = global_node if previous is None else max(previous, global_node)
        self._event(
            "switch_progress_updated" if accepted else "switch_progress_jump_rejected",
            current_map=self.current_map,
            observed_global_node=global_node,
            previous_progress_node=previous,
            progress_node=self._switch_progress_node,
            max_progress_step=max_step,
        )

    def _switch_due(self, global_node: int) -> bool:
        item = self.shards[self.index]
        if self._pending_index is None:
            return False

        policy = self.switch_policies.get(self.current_map, {})
        transition_start = int(
            policy.get(
                "switch_global_start",
                item.get(
                    "switch_global_start",
                    int(item["core_global_stop_exclusive"]) - self.preload_nodes,
                ),
            )
        )
        progress_ready = not policy or (
            self._switch_progress_node is not None and self._switch_progress_node >= transition_start
        )
        return self._pending_index > self.index and global_node >= transition_start and progress_ready

    def _decorate(self, result: dict, switched: bool = False) -> dict:
        result["_map_name"] = self.current_map
        result["_node_offset"] = self.node_offset
        if result.get("node") is not None:
            result["_global_node"] = int(result["node"]) + self.node_offset
        if switched:
            result["_map_switched"] = True
        return result

    def _index_for_global_node(self, global_node: int) -> int:
        return shard_index_for_global_node(
            self.shards,
            global_node,
            self.min_shard_index,
            self.max_shard_index,
        )

    def locate(self, frame: object) -> dict:
        if self._force_recovery and self._recovery is not None:
            self._recovery_calls += 1
            extractor = self.current if self.current is not None else self._recovery
            query = extractor.extract_query(frame)
            raw = self._recovery.locate_features(query)
            recovered = self._apply_recovery(raw)
            self._event("full_recovery_forced_result", result=self._result_summary(raw), accepted=recovered is not None)
            if recovered is not None:
                self._force_recovery = False
                return recovered
            return self._decorate(self._recovery_rejected_result(raw))
        self._force_recovery = False
        if self.current is None:
            return {"ok": False, "inliers": 0, "reason": "шард не выбран — нажмите «СБРОС ШАРДА»"}
        query = self.current.extract_query(frame)
        result = self.current.locate_features(query)
        self._event(
            "active_shard_result",
            current_map=self.current_map,
            current_index=self.index,
            node_offset=self.node_offset,
            result=self._result_summary(result),
        )
        current_good = result.get("ok") and result.get("inliers", 0) >= self.min_inliers
        if not current_good and self._lost_recovery_enabled:
            recovered, raw_recovery = self._collect_recovery_async()
            if recovered is not None:
                return recovered
            self._start_recovery_async(query)
            if raw_recovery is not None and raw_recovery.get("inliers", 0) > result.get("inliers", 0):
                return self._decorate(self._recovery_rejected_result(raw_recovery))
        if result.get("ok") and result.get("node") is not None:
            observed_global_node = int(result["node"]) + self.node_offset
            self._update_switch_progress(observed_global_node)
            self._schedule(observed_global_node)
        self._collect_preload()

        global_node = result.get("node")
        global_node = None if global_node is None else int(global_node) + self.node_offset
        if self._pending is not None:
            candidate = self._pending.locate_features(query)
            policy = self.switch_policies.get(self.current_map, {})
            candidate_min_inliers = int(policy.get("min_candidate_inliers", self.min_inliers))
            max_node_disagreement = int(policy.get("max_node_disagreement", self.preload_nodes))
            required_confirm_fixes = int(policy.get("confirm_fixes", self.confirm_fixes))
            candidate_ok = candidate.get("ok") and candidate.get("inliers", 0) >= candidate_min_inliers
            candidate_global = None
            if candidate_ok and global_node is not None and candidate.get("node") is not None:
                candidate_offset = int(self.shards[int(self._pending_index)]["global_node_start"])
                candidate_global = int(candidate["node"]) + candidate_offset
                candidate_ok = abs(candidate_global - global_node) <= max_node_disagreement
            self._pending_good = self._pending_good + 1 if candidate_ok else 0
            switch_due = global_node is not None and self._switch_due(global_node)
            self._event(
                "candidate_shard_result",
                current_map=self.current_map,
                current_index=self.index,
                current_global_node=global_node,
                candidate_map=self.shards[int(self._pending_index)]["map"],
                candidate_index=self._pending_index,
                candidate_global_node=candidate_global,
                candidate_ok=bool(candidate_ok),
                pending_good=self._pending_good,
                required_confirm_fixes=required_confirm_fixes,
                required_candidate_inliers=candidate_min_inliers,
                max_node_disagreement=max_node_disagreement,
                switch_progress_node=self._switch_progress_node,
                switch_due=bool(switch_due),
                result=self._result_summary(candidate),
            )
            if (
                global_node is not None
                and self._switch_due(global_node)
                and self._pending_good >= required_confirm_fixes
            ):
                previous_map = self.current_map
                previous_index = self.index
                self.index = int(self._pending_index)
                self.current = self._pending
                self.current_map = str(self.shards[self.index]["map"])
                self.node_offset = int(self.shards[self.index]["global_node_start"])
                self._clear_pending()
                self._switch_progress_node = candidate_global
                LOGGER.info("активный шард -> %s", self.current_map)
                self._event(
                    "shard_switched",
                    previous_map=previous_map,
                    previous_index=previous_index,
                    current_map=self.current_map,
                    current_index=self.index,
                    current_global_node=global_node,
                    candidate_global_node=candidate_global,
                    confirmations=required_confirm_fixes,
                )
                self._evict_other(self.index)
                if candidate.get("node") is not None:
                    self._last_global_node = int(candidate["node"]) + self.node_offset
                return self._decorate(candidate, switched=True)
        return self._decorate(result)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._recovery_executor.shutdown(wait=False, cancel_futures=True)
        seen: set[int] = set()
        for localizer in self._loaded.values():
            if id(localizer) not in seen and hasattr(localizer, "close"):
                localizer.close()
            seen.add(id(localizer))
        if self._recovery is not None and id(self._recovery) not in seen and hasattr(self._recovery, "close"):
            self._recovery.close()

    @property
    def preload_status(self) -> dict[str, object]:
        target = None if self._pending_index is None else str(self.shards[self._pending_index]["map"])
        return {
            "target": target,
            "ready": self._pending is not None,
            "loaded": len(self._loaded),
            "total": self.max_shard_index - self.min_shard_index + 1,
            "recovery_map": self.recovery_map if self._recovery is not None else None,
            "recovery_calls": self._recovery_calls,
            "recovery_fixes": self._recovery_fixes,
        }
