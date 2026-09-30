from types import SimpleNamespace

import numpy as np

from module2_localization.core import route_follower


def test_centered_speed_signal_stores_only_magnitude():
    route_follower.set_speed_signal(1500, neutral=1500, deadzone=30)
    assert route_follower._speed_magnitude == 0
    route_follower.set_speed_signal(1531, neutral=1500, deadzone=30)
    assert route_follower._speed_magnitude == 1
    route_follower.set_speed_signal(1469, neutral=1500, deadzone=30)
    assert route_follower._speed_magnitude == 1


def test_speed_lookahead_can_be_disabled():
    follower = SimpleNamespace(
        route=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]]),
        route_fwd=np.array([[0.0, 0.0, 1.0]] * 3),
        route_cum=np.array([0.0, 1.0, 2.0]),
        node_step=1.0,
        lookahead=1,
        lookahead_min=1,
        lookahead_adapt=0.0,
        lookahead_speed_div=1.0,
        lookahead_max=20.0,
        speed_lookahead_enabled=False,
        steer="pursuit",
        stanley_k=1.0,
        heading_gate=0.3,
        deadzone=4.0,
        stop_end_nodes=-1,
    )
    route_follower.set_speed_signal(1800, neutral=1500, deadzone=30)
    result = route_follower.RouteFollower.command(follower, np.zeros(3), np.array([0.0, 0.0, 1.0]))
    assert result["target_node"] == 1
