"""Browser dashboard for localization status and NATS controls on Jetson."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Protocol
from urllib.parse import parse_qs, quote, urlsplit

from .. import config
from .admin_data import log_catalog, map_catalog, map_payload, resolve_log
from .admin_page import HTML
from .nats_gateway import NatsClient

LOGGER = logging.getLogger("localization-admin")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOGS_DIR = PROJECT_ROOT / "logs"


class NatsMessage(Protocol):
    data: bytes


async def serve(bind="0.0.0.0", port=8080, nats_url=None, logs_dir=DEFAULT_LOGS_DIR) -> int:
    logs_dir = Path(logs_dir).expanduser().resolve()

    catalog = map_catalog()
    allowed_maps = {str(item["name"]) for item in catalog}
    payload_cache: dict[str, dict[str, object]] = {}
    latest: dict[str, object] = {}
    last_message: float | None = None

    async def on_status(msg: NatsMessage) -> None:
        nonlocal latest, last_message
        try:
            latest = json.loads(msg.data.decode())
            last_message = time.monotonic()
        except (UnicodeDecodeError, json.JSONDecodeError):
            return

    nc = NatsClient(nats_url or config.NATS_URL)
    await nc.connect()
    await nc.subscribe(config.NATS_TOPIC, on_status)

    async def respond(writer: asyncio.StreamWriter, status: str, body: bytes, content_type: str) -> None:
        writer.write(
            f"HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {len(body)}\r\n"
            "Cache-Control: no-store\r\nConnection: close\r\n\r\n".encode()
            + body
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    async def respond_file(writer: asyncio.StreamWriter, path: Path) -> None:
        size = path.stat().st_size
        encoded_name = quote(path.name, safe="")
        writer.write(
            f"HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\n"
            f"Content-Length: {size}\r\n"
            f"Content-Disposition: attachment; filename*=UTF-8''{encoded_name}\r\n"
            "Cache-Control: no-store\r\nConnection: close\r\n\r\n".encode()
        )
        await writer.drain()
        with path.open("rb") as source:
            while chunk := await asyncio.to_thread(source.read, 1024 * 1024):
                writer.write(chunk)
                await writer.drain()
        writer.close()
        await writer.wait_closed()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            header = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=3)
            lines = header.decode("latin1").split("\r\n")
            method, target, _ = lines[0].split(" ", 2)
            headers = {}
            for line in lines[1:]:
                if ":" in line:
                    key, value = line.split(":", 1)
                    headers[key.lower()] = value.strip()
            length = int(headers.get("content-length", "0"))
            body = await reader.readexactly(length) if length else b""
            path = urlsplit(target).path
            if method == "GET" and path == "/":
                await respond(writer, "200 OK", HTML.encode(), "text/html; charset=utf-8")
            elif method == "GET" and path == "/api/maps":
                await respond(writer, "200 OK", json.dumps({"maps": catalog}).encode(), "application/json")
            elif method == "GET" and path == "/api/routes":
                routes = {key: str(spec.get("label", key)) for key, spec in config.ROUTES.items()}
                await respond(
                    writer,
                    "200 OK",
                    json.dumps({"routes": routes}, ensure_ascii=False).encode(),
                    "application/json; charset=utf-8",
                )
            elif method == "GET" and path == "/api/logs":
                payload = json.dumps({"logs": log_catalog(logs_dir)}, ensure_ascii=False).encode()
                await respond(writer, "200 OK", payload, "application/json; charset=utf-8")
            elif method == "GET" and path == "/api/logs/download":
                name = parse_qs(urlsplit(target).query).get("name", [""])[0]
                log_path = resolve_log(logs_dir, name)
                if log_path is None:
                    await respond(writer, "404 Not Found", b"unknown log", "text/plain")
                    return
                await respond_file(writer, log_path)
            elif method == "GET" and path == "/api/map":
                name = parse_qs(urlsplit(target).query).get("name", [""])[0]
                if name not in allowed_maps:
                    await respond(writer, "404 Not Found", b"unknown map", "text/plain")
                    return
                if name not in payload_cache:
                    payload_cache[name] = map_payload(name)
                await respond(writer, "200 OK", json.dumps(payload_cache[name]).encode(), "application/json")
            elif method == "GET" and path == "/api/status":
                payload = dict(latest)
                payload["nats"] = bool(nc.nc and nc.nc.is_connected)
                payload["age_s"] = None if last_message is None else round(time.monotonic() - last_message, 3)
                await respond(writer, "200 OK", json.dumps(payload).encode(), "application/json")
            elif method == "POST" and path == "/api/control":
                command = json.loads(body or b"{}")
                allowed = {
                    "pause",
                    "resume",
                    "reset",
                    "reset_shard",
                    "reset_traffic",
                    "set_mode",
                    "set_route",
                    "clear_route",
                    "set_traffic",
                    "set_direction",
                    "set_terminal_maneuvers",
                    "set_frame_guard",
                    "set_frame_hash",
                }
                if command.get("cmd") not in allowed:
                    await respond(writer, "400 Bad Request", b'{"message":"unknown command"}', "application/json")
                    return
                if command.get("cmd") == "set_route" and command.get("route") not in config.ROUTES:
                    await respond(writer, "400 Bad Request", b'{"message":"invalid route"}', "application/json")
                    return
                await nc.publish(config.NATS_CONTROL_TOPIC, json.dumps(command).encode())
                answer = json.dumps({"message": f"отправлено: {command['cmd']}"}, ensure_ascii=False).encode()
                await respond(writer, "200 OK", answer, "application/json; charset=utf-8")
            else:
                await respond(writer, "404 Not Found", b"not found", "text/plain")
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ValueError, json.JSONDecodeError):
            if not writer.is_closing():
                await respond(writer, "400 Bad Request", b"bad request", "text/plain")

    server = await asyncio.start_server(handle, bind, port)
    addresses = ", ".join(str(sock.getsockname()) for sock in server.sockets or [])
    LOGGER.info("[admin] http://%s:%d (%s)", bind, port, addresses)
    try:
        async with server:
            await server.serve_forever()
    finally:
        await nc.close()
    return 0
