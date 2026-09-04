import unittest
from types import SimpleNamespace

import numpy as np

from module2_localization.core.camera_navigator import CameraNavigator
from module2_localization.core.command_filter import NavigationCommandFilter, to_command
from module2_localization.core.route_follower import RouteFollower
from module2_localization.runtime.admin_page import HTML
from module2_localization.runtime.frame_timeout import FrameTimeoutGuard
from module2_localization.runtime.motion_controllers import DirectionController, RouteProfileController
from module2_localization.runtime.traffic_controller import TrafficBranch


def pilot_config(**overrides):
    values = {
        "POSE_HISTORY": 5,
        "MIN_NODE_JUMP": 3,
        "MAX_NODES_PER_SEC": 10,
        "MAX_REJECTS": 3,
        "MOVE_EPS": 0.01,
        "STOP_MIN_INLIERS": 20,
        "STOP_CONFIRM": 2,
        "MIN_INLIERS": 15,
        "NAV_MODE": "front",
        "MAPS_DIR": ".",
        "ROUTES": {},
        "DUAL_LOST_HOLD": 2,
        "DUAL_BACK_HOLD": 2,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def localization_result(node=1, move_type="straight", bearing_deg=0.0, inliers=30):
    return {
        "ok": True,
        "node": node,
        "target_node": node + 1,
        "move_type": move_type,
        "bearing_deg": bearing_deg,
        "inliers": inliers,
        "C": np.array([float(node), 0.0, 0.0]),
        "fwd": np.array([1.0, 0.0, 0.0]),
    }


class AdminPageTests(unittest.TestCase):
    def test_route_catalog_is_available_without_telemetry_gate(self):
        self.assertIn('id="showRoutes"', HTML)
        self.assertIn('id="routeCatalog"', HTML)
        self.assertNotIn('id="showRoutes" class="gated"', HTML)


class CommandConversionTests(unittest.TestCase):
    def test_rejects_result_below_inlier_threshold(self):
        command = to_command(localization_result(inliers=14), min_inliers=15)
        self.assertEqual(command, {"move_type": "lost", "reason": "inliers: 14", "inliers": 14})

    def test_preserves_accepted_steering_command(self):
        command = to_command(localization_result(move_type="right", bearing_deg=12.345, inliers=20), 15)
        self.assertEqual(command["move_type"], "right")
        self.assertEqual(command["deg"], 12.35)
        self.assertEqual(command["node"], 1)
        self.assertEqual(command["target_node"], 2)


class PilotBehaviorTests(unittest.TestCase):
    def test_starts_paused_and_reports_operator_pause(self):
        pilot = NavigationCommandFilter(pilot_config())
        command = pilot.step(localization_result())
        self.assertEqual(command["move_type"], "stop")
        self.assertEqual(command["reason"], "operator_paused")
        self.assertTrue(command["paused"])

    def test_latches_route_complete_after_confirmed_stops(self):
        pilot = NavigationCommandFilter(pilot_config())
        pilot.resume()
        pilot.step(localization_result(node=10, move_type="stop"), now=1.0)
        command = pilot.step(localization_result(node=10, move_type="stop"), now=1.1)
        self.assertEqual(command["move_type"], "stop")
        self.assertEqual(command["reason"], "route_complete")
        self.assertTrue(pilot.stopped)

    def test_rejects_large_forward_node_jump(self):
        pilot = NavigationCommandFilter(pilot_config())
        pilot.resume()
        pilot.step(localization_result(node=10), now=1.0)
        command = pilot.step(localization_result(node=30), now=1.1)
        self.assertEqual(command["move_type"], "lost")
        self.assertIn("скачок", command["reason"])


class RouteFollowerBehaviorTests(unittest.TestCase):
    def test_straight_route_produces_straight_command(self):
        follower = SimpleNamespace(
            route=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
            route_fwd=np.array([[1.0, 0.0, 0.0]] * 4),
            route_cum=np.array([0.0, 1.0, 2.0, 3.0]),
            node_step=1.0,
            lookahead=1,
            lookahead_min=1,
            lookahead_adapt=0.0,
            deadzone=4.0,
            heading_gate=0.3,
            stop_end_nodes=0,
            steer="pursuit",
            lookahead_speed_div=None,
        )
        command = RouteFollower.command(
            follower,
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
        )
        self.assertEqual(command["move_type"], "straight")
        self.assertAlmostEqual(command["bearing_deg"], 0.0)
        self.assertEqual(command["node"], 0)
        self.assertEqual(command["target_node"], 1)


class FrameTimeoutGuardTests(unittest.TestCase):
    def test_timeout_and_first_fresh_frame_recovery(self):
        guard = FrameTimeoutGuard(enabled=True, timeout_s=0.5, recovery_frames=1)
        guard.reset(now=0.0)
        self.assertFalse(guard.observe(1, now=0.0)["timed_out"])
        self.assertTrue(guard.observe(1, now=0.5)["timed_out"])
        recovered = guard.observe(2, now=0.6)
        self.assertTrue(recovered["fresh"])
        self.assertFalse(recovered["timed_out"])


class NavigationCoordinatorTests(unittest.TestCase):
    def test_without_route_returns_safe_stop(self):
        navigator = CameraNavigator(None, None, pilot_config(), route=None)
        command = navigator.step(None, None)
        self.assertEqual(command["move_type"], "stop")
        self.assertEqual(command["reason"], "route_not_selected")
        self.assertEqual(command["mode"], "idle")


class RuntimeControllerTests(unittest.TestCase):
    def test_backward_left_guard_changes_only_left_to_right(self):
        cfg = SimpleNamespace(
            BACKWARD_ZONES={"map": ((0, 26),)},
            BACKWARD_LEFT_BLOCK_ZONES={"map": ((0, 26),)},
            STEERING_OUTLIER_GUARDS={},
            DEADZONE_DEG=4.0,
        )
        controller = DirectionController(cfg, enabled=True)
        command = {"map": "map", "global_node": 10, "move_type": "left", "deg": -7.0}
        controller.process(command)
        self.assertEqual(command["direction"], "backward")
        self.assertEqual(command["move_type"], "right")
        self.assertEqual(command["deg"], 173.0)

    def test_route_profile_latches_route_complete(self):
        cfg = SimpleNamespace(
            ROUTE_21_TERMINAL_MANEUVERS_DEFAULT=False,
            ROUTE_21_NO_MANEUVERS_MIN_NODE=27,
            ROUTE_21_NO_MANEUVERS_STOP_NODE=1595,
        )
        controller = RouteProfileController(cfg)
        command = {"route": "2-1", "global_node": 1595, "move_type": "left", "deg": -10.0}
        controller.process(command)
        self.assertEqual(command["move_type"], "stop")
        self.assertEqual(command["reason"], "route_complete")
        next_command = {"route": "2-1", "global_node": None, "move_type": "lost"}
        controller.process(next_command)
        self.assertEqual(next_command["reason"], "route_complete")

    def test_traffic_requires_red_before_green_and_latches_go(self):
        branch = TrafficBranch.__new__(TrafficBranch)
        branch.state = "WAIT_RED"
        branch.completed = False
        branch.last_map = None
        branch.last_node = None
        signals = iter(("green", "red", "green"))
        branch._analyze = lambda image: {"signal": next(signals)}
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        branch.update(True, frame)
        self.assertEqual(branch.state, "WAIT_RED")
        branch.update(True, frame)
        self.assertEqual(branch.state, "WAIT_GREEN")
        branch.update(True, frame)
        self.assertEqual(branch.state, "GO")
        self.assertTrue(branch.go)


if __name__ == "__main__":
    unittest.main()
