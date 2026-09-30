import argparse
import asyncio

from .. import config


def parse_args():
    parser = argparse.ArgumentParser(description="демон локализации: один видеопоток -> команда -> терминал+NATS")
    parser.add_argument("--camera", default=None, help="переопределить источник (индекс/GStreamer)")
    parser.add_argument("--video", default=None, help="видеофайл вместо камеры")
    parser.add_argument("--source-step", type=int, default=2, help="файл: каждый N-й кадр в буфер")
    parser.add_argument(
        "--shm",
        nargs="?",
        const=config.CAM_SHM_SOCKET,
        default=None,
        help=f"кадры из fan-out (по умолчанию {config.CAM_SHM_SOCKET})",
    )
    parser.add_argument(
        "--video-loop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="зациклить видеофайл (по умолчанию включено)",
    )
    parser.add_argument(
        "--route",
        default=None,
        choices=list(config.ROUTES),
        help=f"стартовый маршрут (по умолч. config.DEFAULT_ROUTE={config.DEFAULT_ROUTE!r})",
    )
    parser.add_argument("--no-nats", action="store_true", help="только терминал, без NATS")
    parser.add_argument(
        "--no-recovery",
        action="store_true",
        help="выключить постоянные попытки полной релокализации при LOST",
    )
    return parser.parse_args()


def main():
    from ..runtime.application import run
    return asyncio.run(run(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
