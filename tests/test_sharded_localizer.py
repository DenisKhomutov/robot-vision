import json
import tempfile
import unittest
from pathlib import Path

from module2_localization.services.sharded_localizer import ShardedLocalizer


class FakeBackend:
    def __init__(self, map_name, results):
        self.map_name = map_name
        self.results = results
        self.closed = False

    def extract_query(self, frame):
        return {"frame": frame}

    def locate_features(self, query):
        result = self.results[self.map_name]
        return dict(result() if callable(result) else result)

    def locate(self, frame):
        return self.locate_features(self.extract_query(frame))

    def close(self):
        self.closed = True


class ShardedLocalizerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.maps_dir = Path(self.temp.name)
        shard_root = self.maps_dir / "route" / "front_shard"
        shard_root.mkdir(parents=True)
        self.shards = [
            {
                "map": "route/front_shard/01_of_02",
                "source_map": "route/front_full",
                "global_node_start": 0,
                "core_global_stop_exclusive": 100,
                "preload_global_start": 75,
                "switch_global_start": 90,
            },
            {
                "map": "route/front_shard/02_of_02",
                "source_map": "route/front_full",
                "global_node_start": 90,
                "core_global_stop_exclusive": 200,
                "preload_global_start": 175,
                "switch_global_start": 190,
            },
        ]
        (shard_root / "manifest.json").write_text(json.dumps({"shards": self.shards}))
        full = self.maps_dir / "route" / "front_full"
        full.mkdir(parents=True)
        (full / "runtime.npz").touch()
        (full / "aliked_bank.npz").touch()

    def tearDown(self):
        self.temp.cleanup()

    def make_localizer(self, results, **kwargs):
        created = {}

        def factory(map_name, back_facing):
            backend = FakeBackend(map_name, results)
            created.setdefault(map_name, []).append(backend)
            return backend

        localizer = ShardedLocalizer(
            self.maps_dir,
            self.shards[0]["map"],
            False,
            factory,
            preload_nodes=25,
            confirm_fixes=2,
            min_inliers=15,
            preload_all=True,
            recovery_min_inliers=20,
            **kwargs,
        )
        return localizer, created

    def test_forced_recovery_selects_shard_for_global_node(self):
        results = {
            "route/front_full": {"ok": True, "node": 120, "target_node": 121, "inliers": 30},
            self.shards[0]["map"]: {"ok": True, "node": 10, "inliers": 30},
            self.shards[1]["map"]: {"ok": True, "node": 30, "inliers": 30},
        }
        localizer, _ = self.make_localizer(results)
        localizer.force_relocate()
        result = localizer.locate(object())
        self.assertEqual(localizer.index, 1)
        self.assertEqual(result["node"], 30)
        self.assertEqual(result["target_node"], 31)
        self.assertTrue(result["_full_map_recovery"])
        self.assertTrue(result["_map_switched"])
        localizer.close()

    def test_switches_only_after_required_candidate_confirmations(self):
        results = {
            "route/front_full": {"ok": False, "inliers": 0},
            self.shards[0]["map"]: {"ok": True, "node": 95, "inliers": 30},
            self.shards[1]["map"]: {"ok": True, "node": 5, "inliers": 30},
        }
        localizer, _ = self.make_localizer(results)
        first = localizer.locate(object())
        self.assertEqual(localizer.index, 0)
        self.assertNotIn("_map_switched", first)
        second = localizer.locate(object())
        self.assertEqual(localizer.index, 1)
        self.assertTrue(second["_map_switched"])
        self.assertEqual(second["_global_node"], 95)
        localizer.close()


if __name__ == "__main__":
    unittest.main()
