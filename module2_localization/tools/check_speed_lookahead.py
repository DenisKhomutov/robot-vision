from __future__ import annotations

import argparse
import asyncio
import json

from module2_localization import config


def calculate(speed_pwm: float, offset: float, node_step: float | None) -> dict[str, float | None]:
    minimum = float(config.LOOKAHEAD_MIN)
    maximum = float(config.LOOKAHEAD_MAX)
    divisor = float(config.LOOKAHEAD_SPEED_DIV)
    adaptation = float(config.LOOKAHEAD_ADAPT)
    base = max(minimum, min(float(speed_pwm) / divisor, maximum))
    effective = max(base - adaptation * abs(float(offset)), minimum)
    distance = None if node_step is None else effective * float(node_step)
    return {"base_nodes": base, "effective_nodes": effective, "target_distance": distance}


def print_value(speed_pwm: float, offset: float, node_step: float | None) -> None:
    value = calculate(speed_pwm, offset, node_step)
    distance = "—" if value["target_distance"] is None else f"{value['target_distance']:.4f}"
    print(
        f"PWM={speed_pwm:7.2f}  "
        f"base={value['base_nodes']:6.3f} nodes  "
        f"effective={value['effective_nodes']:6.3f} nodes  "
        f"target_distance={distance}",
        flush=True,
    )


def synthetic(samples: int, offset: float, node_step: float | None) -> None:
    if samples < 2:
        raise SystemExit("samples должен быть >= 2")
    print(
        f"formula: base=clamp(PWM/{config.LOOKAHEAD_SPEED_DIV}, "
        f"{config.LOOKAHEAD_MIN}, {config.LOOKAHEAD_MAX}); "
        f"effective=max(base-{config.LOOKAHEAD_ADAPT}*abs(offset), {config.LOOKAHEAD_MIN})"
    )
    print(f"samples={samples} offset={offset} node_step={node_step}")
    for index in range(samples):
        print_value(255.0 * index / (samples - 1), offset, node_step)


async def listen(seconds: float, offset: float, node_step: float | None) -> None:
    import nats

    nc = await nats.connect(config.NATS_URL)
    count = 0

    async def callback(msg) -> None:
        nonlocal count
        count += 1
        try:
            data = json.loads(msg.data.decode())
            speed_pwm = data.get("speed_pwm")
        except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
            print(f"[{count}] invalid payload={msg.data!r}", flush=True)
            return
        if speed_pwm is None:
            print(f"[{count}] no speed_pwm payload={data!r}", flush=True)
            return
        print(f"[{count}] topic={msg.subject} payload={data!r}", flush=True)
        print_value(float(speed_pwm), offset, node_step)

    await nc.subscribe(config.NATS_SPEED_TOPIC, cb=callback)
    print(f"listening topic={config.NATS_SPEED_TOPIC} url={config.NATS_URL} seconds={seconds}", flush=True)
    await asyncio.sleep(seconds)
    await nc.close()
    print(f"received={count}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--offset", type=float, default=0.0)
    parser.add_argument("--node-step", type=float, default=None)
    parser.add_argument("--listen", type=float, default=0.0)
    args = parser.parse_args()
    synthetic(args.samples, args.offset, args.node_step)
    if args.listen > 0:
        asyncio.run(listen(args.listen, args.offset, args.node_step))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
