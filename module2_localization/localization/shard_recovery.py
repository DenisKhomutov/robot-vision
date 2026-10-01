import logging

LOGGER = logging.getLogger("sharded-localizer")


class ShardRecoveryMixin:
    def force_relocate(self) -> None:
        self._force_recovery = True
        self._event("full_recovery_forced", current_map=self.current_map, current_index=self.index)

    def _start_recovery_async(self, query: object) -> None:
        if self._recovery is None:
            return
        if self._recovery_future is not None:
            return
        self._recovery_calls += 1
        self._event(
            "full_recovery_async_started",
            current_map=self.current_map,
            current_index=self.index,
            recovery_call=self._recovery_calls,
        )
        self._recovery_future = self._recovery_executor.submit(
            self._recovery.locate_features,
            query,
        )

    def _collect_recovery_async(self) -> tuple[dict | None, dict | None]:
        if self._recovery_future is None or not self._recovery_future.done():
            return None, None
        future, self._recovery_future = self._recovery_future, None
        try:
            raw = future.result()
            self._event("full_recovery_async_result", result=self._result_summary(raw))
            return self._apply_recovery(raw), raw
        except Exception as exc:
            LOGGER.error("фоновый recovery упал: %s", exc)
            self._event("full_recovery_async_failed", error=repr(exc))
            return None, None

    def _recovery_rejected_result(self, recovered: dict | None) -> dict:
        recovered = recovered or {}
        inliers = int(recovered.get("inliers", 0))
        result = {
            "ok": False,
            "inliers": inliers,
            "reason": recovered.get("reason") or f"recovery: inliers {inliers}/{self.recovery_min_inliers}",
        }
        pairs = recovered.get("n_pairs", recovered.get("pairs"))
        if pairs is not None:
            result["n_pairs"] = int(pairs)
        if recovered.get("node") is not None:
            result["recovery_candidate_node"] = int(recovered["node"])
        return result

    def _apply_recovery(self, recovered: dict) -> dict | None:
        if not recovered.get("ok") or recovered.get("inliers", 0) < self.recovery_min_inliers:
            self._event(
                "full_recovery_rejected",
                result=self._result_summary(recovered),
                required_inliers=self.recovery_min_inliers,
            )
            return None
        global_node = recovered.get("node")
        if global_node is None:
            self._event("full_recovery_rejected", result=self._result_summary(recovered), reason="node_is_none")
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
        self._clear_pending()
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
        LOGGER.warning(
            "LOST восстановлен полной картой %s: global_node=%d -> %s", self.recovery_map, global_node, self.current_map
        )
        self._event(
            "full_recovery_applied",
            recovery_map=self.recovery_map,
            global_node=global_node,
            target_index=target,
            target_map=self.current_map,
            switched=switched,
            result=self._result_summary(recovered),
        )
        return recovered
