"""Browser dashboard for localization status and NATS controls on Jetson."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from typing import Protocol
from urllib.parse import urlsplit

import numpy as np

from . import config
from .nats_client import NatsClient

LOGGER = logging.getLogger("localization-admin")


class NatsMessage(Protocol):
    data: bytes


HTML = r"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Robot Vision</title><style>
:root{color-scheme:dark;font-family:system-ui,sans-serif}body{margin:0;background:#101216;color:#eee}
header{display:flex;gap:12px;align-items:center;padding:12px 16px;background:#191d24;flex-wrap:wrap}
.badge{padding:5px 10px;border-radius:16px;background:#303641}.ok{background:#174f31}.bad{background:#682525}
main{display:grid;grid-template-columns:minmax(300px,1fr) 320px;gap:12px;padding:12px;max-width:1300px;margin:auto}
canvas{width:100%;height:auto;background:#111;border:1px solid #343a44;border-radius:8px}
.panel{background:#191d24;border-radius:8px;padding:14px}.value{font-size:1.35rem;font-weight:700;margin:4px 0 14px}
.controls{display:grid;grid-template-columns:1fr 1fr;gap:10px}button{font-size:1rem;padding:14px;border:0;border-radius:8px;background:#344054;color:#fff}
button.go{background:#18794e}button.stop{background:#a33b32}button:active{transform:scale(.98)}
#message{min-height:1.5em;color:#f5c451}@media(max-width:800px){main{grid-template-columns:1fr}.panel{order:-1}}
</style></head><body><header><strong>Robot Vision</strong><span id="link" class="badge bad">NATS</span>
<span id="fresh" class="badge bad">нет телеметрии</span><span id="mode" class="badge">—</span></header>
<main><canvas id="map" width="900" height="700"></canvas><section class="panel">
<div>Команда</div><div id="command" class="value">LOST</div><div>Узел</div><div id="node" class="value">—</div>
<div>Отклонение</div><div id="offset" class="value">—</div><div>Инлайнеры</div><div id="inliers" class="value">—</div>
<div class="controls"><button class="go" data-cmd="resume">СТАРТ</button><button class="stop" data-cmd="pause">ПАУЗА</button>
<button data-cmd="reset">СБРОС</button><button data-cmd="set_mode" data-mode="rear">REAR</button>
<button data-cmd="set_mode" data-mode="dual">DUAL</button></div><p id="message"></p></section></main>
<script>
let maps={},status={}; const canvas=document.querySelector('#map'),ctx=canvas.getContext('2d');
function fit(points){let xs=points.map(p=>p[0]),ys=points.map(p=>p[1]),lo=[Math.min(...xs),Math.min(...ys)],hi=[Math.max(...xs),Math.max(...ys)];
 return p=>[35+(p[0]-lo[0])/(hi[0]-lo[0]||1)*(canvas.width-70),canvas.height-35-(p[1]-lo[1])/(hi[1]-lo[1]||1)*(canvas.height-70)]}
function draw(){let key=status.cam==='front'?'front':'rear',m=maps[key]||maps.rear;if(!m)return;let all=m.cloud.length?m.cloud:m.route,px=fit(all);
 ctx.fillStyle='#101216';ctx.fillRect(0,0,canvas.width,canvas.height);ctx.fillStyle='#555';for(let p of m.cloud){let q=px(p);ctx.fillRect(q[0],q[1],1,1)}
 ctx.strokeStyle='#d4ae45';ctx.lineWidth=3;ctx.beginPath();m.route.forEach((p,i)=>{let q=px(p);i?ctx.lineTo(...q):ctx.moveTo(...q)});ctx.stroke();
 if(status.pos){let q=px(status.pos);ctx.fillStyle='#ff4b4b';ctx.beginPath();ctx.arc(q[0],q[1],8,0,7);ctx.fill();if(status.head){let h=px([status.pos[0]+status.head[0]*.5,status.pos[1]+status.head[1]*.5]);ctx.strokeStyle='#fff';ctx.beginPath();ctx.moveTo(...q);ctx.lineTo(...h);ctx.stroke()}}
 ctx.fillStyle='#eee';ctx.font='18px system-ui';ctx.fillText(m.name,18,26)}
async function tick(){try{let r=await fetch('/api/status',{cache:'no-store'});status=await r.json();document.querySelector('#link').textContent=status.nats?'NATS подключён':'NATS отключён';document.querySelector('#link').className='badge '+(status.nats?'ok':'bad');
 let age=status.age_s;document.querySelector('#fresh').textContent=age==null?'нет телеметрии':`телеметрия ${age.toFixed(1)}с`;document.querySelector('#fresh').className='badge '+(age!=null&&age<2?'ok':'bad');
 document.querySelector('#mode').textContent=(status.mode||'—')+' '+(status.cam||'');document.querySelector('#command').textContent=(status.move_type||'LOST').toUpperCase()+(status.deg!=null?` ${status.deg>0?'+':''}${status.deg}°`:'');
 document.querySelector('#node').textContent=status.node==null?'—':status.node;document.querySelector('#offset').textContent=status.dist_to_route_m!=null?status.dist_to_route_m+' м':(status.dist_to_route??'—');document.querySelector('#inliers').textContent=status.inliers??'—';draw()}catch(e){document.querySelector('#message').textContent=e}setTimeout(tick,250)}
document.querySelectorAll('button').forEach(b=>b.onclick=async()=>{let body={cmd:b.dataset.cmd};if(b.dataset.mode)body.mode=b.dataset.mode;let r=await fetch('/api/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});document.querySelector('#message').textContent=(await r.json()).message});
fetch('/api/maps').then(r=>r.json()).then(x=>{maps=x;draw()});tick();
</script></body></html>"""


def map_payload(name: str, cloud_limit: int = 12_000) -> dict[str, object]:
    path = config.MAPS_DIR / name / "runtime.npz"
    data = np.load(path)
    pos = data["pos"][:, [0, 2]]
    points = data["points"][:, [0, 2]]
    if len(points) > cloud_limit:
        indices = np.linspace(0, len(points) - 1, cloud_limit, dtype=int)
        points = points[indices]
    return {"name": name, "route": pos.round(4).tolist(), "cloud": points.round(4).tolist()}


async def main() -> int:
    parser = argparse.ArgumentParser(description="веб-админка локализации")
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--nats-url", default=config.NATS_URL)
    parser.add_argument("--rear-map", default=config.REAR_MAP)
    parser.add_argument("--front-map", default=config.FRONT_MAP)
    args = parser.parse_args()

    maps = {"rear": map_payload(args.rear_map)}
    if args.front_map != args.rear_map:
        maps["front"] = map_payload(args.front_map)
    latest: dict[str, object] = {}
    last_message: float | None = None

    async def on_status(msg: NatsMessage) -> None:
        nonlocal latest, last_message
        try:
            latest = json.loads(msg.data.decode())
            last_message = time.monotonic()
        except (UnicodeDecodeError, json.JSONDecodeError):
            return

    nc = NatsClient(args.nats_url)
    await nc.connect()
    await nc.subscribe(config.NATS_TOPIC, on_status)

    async def respond(writer: asyncio.StreamWriter, status: str, body: bytes, content_type: str) -> None:
        writer.write(
            f"HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {len(body)}\r\n"
            "Cache-Control: no-store\r\nConnection: close\r\n\r\n".encode() + body
        )
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
                await respond(writer, "200 OK", json.dumps(maps).encode(), "application/json")
            elif method == "GET" and path == "/api/status":
                payload = dict(latest)
                payload["nats"] = bool(nc.nc and nc.nc.is_connected)
                payload["age_s"] = None if last_message is None else round(time.monotonic() - last_message, 3)
                await respond(writer, "200 OK", json.dumps(payload).encode(), "application/json")
            elif method == "POST" and path == "/api/control":
                command = json.loads(body or b"{}")
                allowed = {"pause", "resume", "reset", "set_mode"}
                if command.get("cmd") not in allowed:
                    await respond(writer, "400 Bad Request", b'{"message":"unknown command"}', "application/json")
                    return
                await nc.publish(config.NATS_CONTROL_TOPIC, json.dumps(command).encode())
                answer = json.dumps({"message": f"отправлено: {command['cmd']}"}, ensure_ascii=False).encode()
                await respond(writer, "200 OK", answer, "application/json; charset=utf-8")
            else:
                await respond(writer, "404 Not Found", b"not found", "text/plain")
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ValueError, json.JSONDecodeError):
            if not writer.is_closing():
                await respond(writer, "400 Bad Request", b"bad request", "text/plain")

    server = await asyncio.start_server(handle, args.bind, args.port)
    addresses = ", ".join(str(sock.getsockname()) for sock in server.sockets or [])
    LOGGER.info("[admin] http://%s:%d (%s)", args.bind, args.port, addresses)
    try:
        async with server:
            await server.serve_forever()
    finally:
        await nc.close()
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(asyncio.run(main()))
