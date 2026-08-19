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
    ) -> None:
        self.maps_dir = maps_dir
        self.back_facing = back_facing
        self.factory = factory
        self.preload_nodes = preload_nodes
        self.confirm_fixes = confirm_fixes
        self.min_inliers = min_inliers
        self.recovery_min_inliers = recovery_min_inliers
        self.shards = self._load_chain(start_map)
        self.index = next(i for i, item in enumerate(self.shards) if item["map"] == start_map)
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
        # full_recovery управляет ТОЛЬКО постоянными попытками при LOST в фоне
        # (locate()). Саму полную карту грузим всегда, если она есть — она нужна
        # для разового выбора шарда на старте/смене маршрута/reset_shard даже
        # с --no-recovery.
        self._lost_recovery_enabled = bool(full_recovery)
        self._recovery_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="map-recovery")
        self._recovery_future: Future | None = None
        self.recovery_map = str(self.shards[0].get("source_map", ""))
        self._recovery_calls = 0
        self._recovery_fixes = 0
        if not self.recovery_map:
            raise RuntimeError("manifest шардов не содержит source_map для recovery")
        required = (self.maps_dir / self.recovery_map / "runtime.npz",
                    self.maps_dir / self.recovery_map / "aliked_bank.npz")
        if not all(path.exists() for path in required):
            raise RuntimeError(f"нет полного runtime-комплекта recovery: {self.recovery_map}")
        started = time.monotonic()
        self._recovery = factory(self.recovery_map, back_facing)
        LOGGER.info("полная recovery-карта %s загружена за %.2fс", self.recovery_map,
                    time.monotonic() - started)
        # Полную карту НЕ гоняем автоматически при создании — только по явной
        # команде оператора (force_relocate(), дёргает кнопка «СБРОС ШАРДА» в
        # админке). До этого locate() просто отдаёт lost, ничего не трогая.
        if preload_all:
            started = time.monotonic()
            for i, item in enumerate(self.shards):
                if i not in self._loaded:
                    LOGGER.info("загрузка шарда в память %s", item["map"])
                    self._loaded[i] = factory(str(item["map"]), back_facing)
            if self.current is None:
                self.current = self._loaded[self.index]
            LOGGER.info("вся цепочка из %d шардов загружена за %.2fс", len(self.shards), time.monotonic() - started)

    def _load_chain(self, start_map: str) -> list[dict]:
        for manifest in self.maps_dir.glob("*_manifest.json"):
            data = json.loads(manifest.read_text())
            shards = data.get("shards", [])
            if any(item.get("map") == start_map for item in shards):
                return shards
        raise RuntimeError(f"для шарда {start_map} не найден manifest")

    def _start_preload(self, target: int) -> None:
        if target < 0 or target >= len(self.shards) or target == self.index:
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
            return
        LOGGER.info("начата фоновая загрузка %s", map_name)
        self._future = self._executor.submit(self.factory, map_name, self.back_facing)

    def force_relocate(self) -> None:
        """Ignore the current shard on the next locate() and relocalize via
        the full recovery map. Call this whenever navigation resumes after a
        pause — the robot may have been moved between shards while stopped,
        and the current shard's local node no longer means anything."""
        self._force_recovery = True

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
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("не удалось предзагрузить шард: %s", exc)
                self._future = None
                self._pending_index = None
                self._preload_started = None

    def _schedule(self, global_node: int) -> None:
        item = self.shards[self.index]
        core_stop = int(item["core_global_stop_exclusive"])
        self._last_global_node = global_node
        # Маршруты проекта направлены A->B. Нельзя выводить направление цепочки
        # из одного шумного PnP-скачка: это уже запускало загрузку старого шарда.
        if global_node >= core_stop - self.preload_nodes:
            self._start_preload(self.index + 1)

    def _switch_due(self, global_node: int) -> bool:
        item = self.shards[self.index]
        if self._pending_index is None:
            return False
        # Переходим внутри физического перекрытия, как только соседняя карта дала
        # confirm_fixes согласованных фикса. Ожидание core-границы оставляло робота
        # на деградирующем старом банке и создавало LOST прямо перед переключением.
        transition_start = int(item["core_global_stop_exclusive"]) - self.preload_nodes
        return self._pending_index > self.index and global_node >= transition_start

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
                return i
        return len(self.shards) - 1

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
        self._recovery_future = self._recovery_executor.submit(self._recovery.locate, frame)

    def _collect_recovery_async(self) -> dict | None:
        if self._recovery_future is None or not self._recovery_future.done():
            return None
        future, self._recovery_future = self._recovery_future, None
        try:
            return self._apply_recovery(future.result())
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("фоновый recovery упал: %s", exc)
            return None

    def _apply_recovery(self, recovered: dict) -> dict | None:
        if not recovered.get("ok") or recovered.get("inliers", 0) < self.recovery_min_inliers:
            return None
        global_node = recovered.get("node")
        if global_node is None:
            return None
        global_node = int(global_node)
        target = self._index_for_global_node(global_node)
        if target not in self._loaded:
            # Обычно недостижимо при preload_all, но recovery не должен зависеть
            # от этой настройки.
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
        # Pilot работает с локальными node текущего шарда. Геометрическая команда
        # полной и shard-карты находится в одной системе координат, поэтому
        # переводим только индексы маршрута.
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
        return recovered

    def locate(self, frame: object) -> dict:
        if self._force_recovery and self._recovery is not None:
            recovered = self._recover(frame)
            if recovered is not None:
                self._force_recovery = False
                return recovered
            return {"ok": False, "inliers": 0, "reason": "resume: релокализация по полной карте"}
        self._force_recovery = False
        if self.current is None:
            return {"ok": False, "inliers": 0, "reason": "шард не выбран — нажмите «СБРОС ШАРДА»"}
        result = self.current.locate(frame)
        current_good = result.get("ok") and result.get("inliers", 0) >= self.min_inliers
        if not current_good and self._lost_recovery_enabled:
            # Не блокируем кадр на тяжёлый поиск по полной карте — запускаем/держим
            # его в фоне и параллельно продолжаем штатно проверять свой шард каждый
            # кадр (он же ниже, через self.current.locate). Берём фоновый результат,
            # как только он готов, а не раньше.
            self._start_recovery_async(frame)
            recovered = self._collect_recovery_async()
            if recovered is not None:
                return recovered
        if result.get("ok") and result.get("node") is not None:
            self._schedule(int(result["node"]) + self.node_offset)
        self._collect_preload()

        global_node = result.get("node")
        global_node = None if global_node is None else int(global_node) + self.node_offset
        if self._pending is not None:
            candidate = self._pending.locate(frame)
            candidate_ok = candidate.get("ok") and candidate.get("inliers", 0) >= self.min_inliers
            if candidate_ok and global_node is not None and candidate.get("node") is not None:
                candidate_offset = int(self.shards[int(self._pending_index)]["global_node_start"])
                candidate_global = int(candidate["node"]) + candidate_offset
                candidate_ok = abs(candidate_global - global_node) <= self.preload_nodes
            self._pending_good = self._pending_good + 1 if candidate_ok else 0
            if (global_node is not None and self._switch_due(global_node)
                    and self._pending_good >= self.confirm_fixes):
                self.index = int(self._pending_index)
                self.current = self._pending
                self.current_map = str(self.shards[self.index]["map"])
                self.node_offset = int(self.shards[self.index]["global_node_start"])
                self._pending = None
                self._pending_index = None
                self._future = None
                self._pending_good = 0
                self._preload_started = None
                LOGGER.info("активный шард -> %s", self.current_map)
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
                "loaded": len(self._loaded), "total": len(self.shards),
                "recovery_map": self.recovery_map if self._recovery is not None else None,
                "recovery_calls": self._recovery_calls, "recovery_fixes": self._recovery_fixes}
