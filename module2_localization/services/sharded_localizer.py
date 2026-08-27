"""Background-preloaded chain of overlapping runtime localization shards."""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Protocol

LOGGER = logging.getLogger("sharded-localizer")


class Localizer(Protocol):
    def locate(self, frame: object) -> dict: ...


class ShardedLocalizer:
    """Keep driving on the current shard while the adjacent shard loads."""

    def __init__(
        self,
        maps_dir: Path,
        start_map: str,
        back_facing: bool,
        factory: Callable[[str, bool], Localizer],
        preload_nodes: int = 25,
        confirm_fixes: int = 2,
        min_inliers: int = 20,
        preload_all: bool = False,
        full_recovery: bool = False,
        recovery_min_inliers: int = 35,
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
        self.shards = self._load_chain(start_map)
        self.min_shard_index = 0 if min_shard_index is None else int(min_shard_index)
        self.max_shard_index = len(self.shards) - 1 if max_shard_index is None else int(max_shard_index)
        if not 0 <= self.min_shard_index <= self.max_shard_index < len(self.shards):
            raise ValueError("недопустимые границы цепочки шардов")
        self.index = next(i for i, item in enumerate(self.shards) if item["map"] == start_map)
        if not self.min_shard_index <= self.index <= self.max_shard_index:
            raise ValueError("стартовый шард находится вне разрешённого диапазона")
        self.current: Localizer | None = None
        self._loaded: dict[int, Localizer] = {}
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
        self.recovery_map = str(self.shards[0].get("source_map", ""))
        self._recovery_calls = 0
        self._recovery_fixes = 0
        self._switch_progress_node: int | None = None
        if not self.recovery_map:
            raise RuntimeError("manifest шардов не содержит source_map для recovery")
        required = (self.maps_dir / self.recovery_map / "runtime.npz",
                    self.maps_dir / self.recovery_map / "aliked_bank.npz")
        if not all(path.exists() for path in required):
            raise RuntimeError(f"нет полного runtime-комплекта recovery: {self.recovery_map}")
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
        LOGGER.info("полная recovery-карта %s загружена за %.2fс", self.recovery_map,
                    time.monotonic() - started)



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
        if self.event_sink is None:
            return
        try:
            self.event_sink(event, component="sharded_localizer", **fields)
        except Exception:
            LOGGER.exception("ошибка записи диагностического события %s", event)

    @staticmethod
    def _result_summary(result: dict | None) -> dict:
        if not result:
            return {}
        keys = (
            "ok", "node", "target_node", "inliers", "pairs", "n_pairs", "reason",
            "move_type", "bearing_deg", "reproj_error", "error", "frame_ms",
            "extract_ms", "match_ms", "t_ext", "t_match", "t_pnp", "dist_to_route",
            "offset", "dist_to_route_m", "offset_m",
        )
        summary = {key: result.get(key) for key in keys if key in result}
        for key in ("C", "fwd"):
            value = result.get(key)
            if value is not None:
                summary[key] = value
        return summary

    def _load_chain(self, start_map: str) -> list[dict]:
        for manifest in self.maps_dir.glob("**/manifest.json"):
            data = json.loads(manifest.read_text())
            shards = data.get("shards", [])
            if any(item.get("map") == start_map for item in shards):
                return shards
        raise RuntimeError(f"для шарда {start_map} не найден manifest")

    def _start_preload(self, target: int) -> None:
        if target < self.min_shard_index or target > self.max_shard_index or target == self.index:
            self._event("shard_preload_skipped", current_index=self.index, target_index=target,
                        reason="outside_allowed_range_or_current")
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
            self._event("shard_preload_ready", map=map_name, target_index=target,
                        source="already_loaded", elapsed_s=0.0)
            return
        LOGGER.info("начата фоновая загрузка %s", map_name)
        self._event("shard_preload_started", map=map_name, current_map=self.current_map,
                    current_index=self.index, target_index=target,
                    last_global_node=self._last_global_node)
        self._future = self._executor.submit(self.factory, map_name, self.back_facing)

    def force_relocate(self) -> None:
        """Ignore the current shard on the next locate() and relocalize via
        the full recovery map. Call this whenever navigation resumes after a
        pause — the robot may have been moved between shards while stopped,
        and the current shard's local node no longer means anything."""
        self._force_recovery = True
        self._event("full_recovery_forced", current_map=self.current_map, current_index=self.index)

    def _evict_other(self, keep_index: int) -> None:
        for i in list(self._loaded):
            if i == keep_index:
                continue
            localizer = self._loaded.pop(i)
            if hasattr(localizer, "close"):
                localizer.close()

    def _collect_preload(self) -> None:
        if self._future and self._future.done() and self._pending is None:
            try:
                self._pending = self._future.result()
                self._loaded[int(self._pending_index)] = self._pending
                elapsed = 0.0 if self._preload_started is None else time.monotonic() - self._preload_started
                LOGGER.info("предзагружен шард %s за %.2fс", self.shards[int(self._pending_index)]["map"], elapsed)
                self._event("shard_preload_ready", map=self.shards[int(self._pending_index)]["map"],
                            target_index=self._pending_index, source="background", elapsed_s=elapsed)
            except Exception as exc:
                LOGGER.error("не удалось предзагрузить шард: %s", exc)
                self._event("shard_preload_failed", target_index=self._pending_index, error=repr(exc))
                self._future = None
                self._pending_index = None
                self._preload_started = None

    def _schedule(self, global_node: int) -> None:
        item = self.shards[self.index]
        core_stop = int(item["core_global_stop_exclusive"])
        self._last_global_node = global_node


        preload_start = int(item.get("preload_global_start", core_stop - self.preload_nodes))
        self._event("shard_schedule_check", current_map=self.current_map,
                    global_node=global_node, preload_start=preload_start,
                    core_stop_exclusive=core_stop, pending_index=self._pending_index)
        if global_node >= preload_start:
            self._start_preload(self.index + 1)

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
        transition_start = int(policy.get("switch_global_start", item.get(
            "switch_global_start",
            int(item["core_global_stop_exclusive"]) - self.preload_nodes,
        )))
        progress_ready = not policy or (
            self._switch_progress_node is not None
            and self._switch_progress_node >= transition_start
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
        for i, item in enumerate(self.shards):
            if global_node < int(item["core_global_stop_exclusive"]):
                return min(max(i, self.min_shard_index), self.max_shard_index)
        return self.max_shard_index

    def _recover(self, frame: object) -> dict | None:
        """Синхронный recovery — блокирует вызывающего. Используется только для
        разового force_relocate() на resume, где это осознанно допустимо."""
        if self._recovery is None:
            return None
        self._recovery_calls += 1
        return self._apply_recovery(self._recovery.locate(frame))

    def _start_recovery_async(self, frame: object) -> None:
        """Запустить/держать recovery по полной карте в фоновом потоке — не
        блокирует основной цикл, шард продолжает проверяться каждый кадр как
        обычно. Пока прошлый запрос не завершился, новый не запускается."""
        if self._recovery is None:
            return
        if self._recovery_future is not None and not self._recovery_future.done():
            return
        self._recovery_calls += 1
        self._event("full_recovery_async_started", current_map=self.current_map,
                    current_index=self.index, recovery_call=self._recovery_calls)
        self._recovery_future = self._recovery_executor.submit(self._recovery.locate, frame)

    def _collect_recovery_async(self) -> dict | None:
        if self._recovery_future is None or not self._recovery_future.done():
            return None
        future, self._recovery_future = self._recovery_future, None
        try:
            result = future.result()
            self._event("full_recovery_async_result", result=self._result_summary(result))
            return self._apply_recovery(result)
        except Exception as exc:
            LOGGER.error("фоновый recovery упал: %s", exc)
            self._event("full_recovery_async_failed", error=repr(exc))
            return None

    def _apply_recovery(self, recovered: dict) -> dict | None:
        if not recovered.get("ok") or recovered.get("inliers", 0) < self.recovery_min_inliers:
            self._event("full_recovery_rejected", result=self._result_summary(recovered),
                        required_inliers=self.recovery_min_inliers)
            return None
        global_node = recovered.get("node")
        if global_node is None:
            self._event("full_recovery_rejected", result=self._result_summary(recovered),
                        reason="node_is_none")
            return None
        global_node = int(global_node)
        target = self._index_for_global_node(global_node)
        if target not in self._loaded:


            name = str(self.shards[target]["map"])
            self._loaded[target] = self.factory(name, self.back_facing)
        switched = target != self.index
        self.index = target
        self.current = self._loaded[target]
        self.current_map = str(self.shards[target]["map"])
        self.node_offset = int(self.shards[target]["global_node_start"])
        self._pending = None
        self._pending_index = None
        self._future = None
        self._pending_good = 0
        self._preload_started = None
        self._last_global_node = global_node
        self._switch_progress_node = global_node



        recovered = dict(recovered)
        recovered["node"] = global_node - self.node_offset
        if recovered.get("target_node") is not None:
            recovered["target_node"] = int(recovered["target_node"]) - self.node_offset
        recovered["_map_name"] = self.current_map
        recovered["_node_offset"] = self.node_offset
        recovered["_full_map_recovery"] = True
        recovered["_recovery_map"] = self.recovery_map
        if switched:
            recovered["_map_switched"] = True
            self._evict_other(self.index)
        self._recovery_fixes += 1
        LOGGER.warning("LOST восстановлен полной картой %s: global_node=%d -> %s",
                       self.recovery_map, global_node, self.current_map)
        self._event("full_recovery_applied", recovery_map=self.recovery_map,
                    global_node=global_node, target_index=target, target_map=self.current_map,
                    switched=switched, result=self._result_summary(recovered))
        return recovered

    def locate(self, frame: object) -> dict:
        if self._force_recovery and self._recovery is not None:
            recovered = self._recover(frame)
            self._event("full_recovery_forced_result", result=self._result_summary(recovered))
            if recovered is not None:
                self._force_recovery = False
                return recovered
            return {"ok": False, "inliers": 0, "reason": "resume: релокализация по полной карте"}
        self._force_recovery = False
        if self.current is None:
            return {"ok": False, "inliers": 0, "reason": "шард не выбран — нажмите «СБРОС ШАРДА»"}
        result = self.current.locate(frame)
        self._event("active_shard_result", current_map=self.current_map,
                    current_index=self.index, node_offset=self.node_offset,
                    result=self._result_summary(result))
        current_good = result.get("ok") and result.get("inliers", 0) >= self.min_inliers
        if not current_good and self._lost_recovery_enabled:




            self._start_recovery_async(frame)
            recovered = self._collect_recovery_async()
            if recovered is not None:
                return recovered
        if result.get("ok") and result.get("node") is not None:
            observed_global_node = int(result["node"]) + self.node_offset
            self._update_switch_progress(observed_global_node)
            self._schedule(observed_global_node)
        self._collect_preload()

        global_node = result.get("node")
        global_node = None if global_node is None else int(global_node) + self.node_offset
        if self._pending is not None:
            candidate = self._pending.locate(frame)
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
            if (global_node is not None and self._switch_due(global_node)
                    and self._pending_good >= required_confirm_fixes):
                previous_map = self.current_map
                previous_index = self.index
                self.index = int(self._pending_index)
                self.current = self._pending
                self.current_map = str(self.shards[self.index]["map"])
                self.node_offset = int(self.shards[self.index]["global_node_start"])
                self._pending = None
                self._pending_index = None
                self._future = None
                self._pending_good = 0
                self._preload_started = None
                self._switch_progress_node = candidate_global
                LOGGER.info("активный шард -> %s", self.current_map)
                self._event("shard_switched", previous_map=previous_map,
                            previous_index=previous_index, current_map=self.current_map,
                            current_index=self.index, current_global_node=global_node,
                            candidate_global_node=candidate_global,
                            confirmations=required_confirm_fixes)
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
        return {"target": target, "ready": self._pending is not None,
                "loaded": len(self._loaded), "total": self.max_shard_index - self.min_shard_index + 1,
                "recovery_map": self.recovery_map if self._recovery is not None else None,
                "recovery_calls": self._recovery_calls, "recovery_fixes": self._recovery_fixes}
