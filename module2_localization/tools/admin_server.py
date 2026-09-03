import argparse
import asyncio
import logging
from pathlib import Path

from .. import config


def parse_args():
    parser = argparse.ArgumentParser(description="веб-админка локализации")
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--nats-url", default=config.NATS_URL)
    parser.add_argument("--logs-dir", type=Path, default=Path(__file__).resolve().parents[2] / "logs")
    return parser.parse_args()


def main():
    from ..runtime.admin_http import serve

    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return asyncio.run(serve(args.bind, args.port, args.nats_url, args.logs_dir))


if __name__ == "__main__":
    raise SystemExit(main())
