import argparse
import asyncio

from .. import config


def parse_args():
    parser = argparse.ArgumentParser(description="демон локализации: камера -> команда -> терминал+NATS")
    parser.add_argument("--camera", default=None, help="переопределить источник (индекс/GStreamer)")
    parser.add_argument("--video", default=None, help="видеофайл вместо камеры (отладка, только зад)")
    parser.add_argument("--source-step", type=int, default=2, help="файл: каждый N-й кадр в буфер")
    parser.add_argument(
        "--shm",
        nargs="?",
        const=config.CAM_SHM_SOCKET,
        default=None,
        help=f"кадры из ветки fan-out (по умолчанию {config.CAM_SHM_SOCKET})",
    )
    parser.add_argument(
        "--dual", action="store_true", help="двухкамерный режим: грузит фронтальный и задний локализаторы"
    )
    parser.add_argument("--video-front", default=None, help="dual: видео передней (отладка)")
    parser.add_argument("--video-rear", default=None, help="dual: видео задней (отладка)")
    parser.add_argument(
        "--video-loop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="зациклить файловые видеопотоки (по умолчанию включено)",
    )
    parser.add_argument(
        "--mode", default=None, choices=["rear", "dual", "front"], help="стартовый режим (по умолч. из маршрута)"
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
        help="выключить ПОСТОЯННЫЕ попытки recovery при LOST (замер чистой "
        "скорости шарда); полная карта всё равно грузится и используется "
        "разово для выбора шарда на старте/смене маршрута/reset_shard",
    )
    return parser.parse_args()


def main():
    from ..runtime.application import run

    return asyncio.run(run(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
