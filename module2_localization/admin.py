"""Browser dashboard for localization status and NATS controls on Jetson."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from typing import Protocol
from urllib.parse import parse_qs, urlsplit

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
<span id="fresh" class="badge bad">нет телеметрии</span><span id="mode" class="badge">—</span>
<span id="traffic" class="badge">Светофор OFF</span><span id="recovery" class="badge">SHARD</span></header>
<main><canvas id="map" width="900" height="700"></canvas><section class="panel">
<div>Команда</div><div id="command" class="value">LOST</div><div>Узел</div><div id="node" class="value">—</div>
<div>Отклонение</div><div id="offset" class="value">—</div><div>Инлайнеры</div><div id="inliers" class="value">—</div>
<div class="controls"><button class="go" data-cmd="resume">СТАРТ</button><button class="stop" data-cmd="pause">ПАУЗА</button>
<button data-cmd="reset">СБРОС</button><button data-cmd="set_mode" data-mode="rear">REAR</button>
<button data-cmd="set_mode" data-mode="dual">DUAL</button>
<button class="go" data-cmd="set_traffic" data-enabled="true">СВЕТОФОР ON</button>
<button data-cmd="set_traffic" data-enabled="false">СВЕТОФОР OFF</button></div><hr>
<label>Маршрут</label><select id="routeSelect"><option value="route12">Маршрут 1-2</option><option value="route3">Маршрут 3</option></select>
<button id="setRoute">ВЫБРАТЬ МАРШРУТ</button><hr>
<label>Камера для карты</label><select id="camera"><option value="front">front</option><option value="rear">rear</option></select>
<label>Полная карта или шард</label><select id="mapSelect"></select><button id="setMap">ЗАГРУЗИТЬ КАРТУ</button>
<p id="message"></p></section></main>
<script>
let maps={},catalog=[],status={}; const canvas=document.querySelector('#map'),ctx=canvas.getContext('2d');
function fit(points){let xs=points.map(p=>p[0]),ys=points.map(p=>p[1]),lo=[Math.min(...xs),Math.min(...ys)],hi=[Math.max(...xs),Math.max(...ys)];
 return p=>[35+(p[0]-lo[0])/(hi[0]-lo[0]||1)*(canvas.width-70),canvas.height-35-(p[1]-lo[1])/(hi[1]-lo[1]||1)*(canvas.height-70)]}
function draw(){let name=status.map||document.querySelector('#mapSelect').value,m=maps[name];if(!m){if(name)ensureMap(name);return}let all=m.cloud.length?m.cloud:m.route,px=fit(all);
 ctx.fillStyle='#101216';ctx.fillRect(0,0,canvas.width,canvas.height);ctx.fillStyle='#555';for(let p of m.cloud){let q=px(p);ctx.fillRect(q[0],q[1],1,1)}
 ctx.strokeStyle='#d4ae45';ctx.lineWidth=3;ctx.beginPath();m.route.forEach((p,i)=>{let q=px(p);i?ctx.lineTo(...q):ctx.moveTo(...q)});ctx.stroke();
 if(status.pos){let q=px(status.pos);ctx.fillStyle='#ff4b4b';ctx.beginPath();ctx.arc(q[0],q[1],8,0,7);ctx.fill();if(status.head){let h=px([status.pos[0]+status.head[0]*.5,status.pos[1]+status.head[1]*.5]);ctx.strokeStyle='#fff';ctx.beginPath();ctx.moveTo(...q);ctx.lineTo(...h);ctx.stroke()}}
 ctx.fillStyle='#eee';ctx.font='18px system-ui';ctx.fillText(m.name,18,26)}
async function ensureMap(name){if(!name||maps[name])return;let r=await fetch('/api/map?name='+encodeURIComponent(name));if(r.ok){maps[name]=await r.json();draw()}}
function fillMaps(){let cam=document.querySelector('#camera').value,sel=document.querySelector('#mapSelect'),old=sel.value;sel.innerHTML='';for(let m of catalog.filter(x=>x.camera===cam)){let o=document.createElement('option');o.value=m.name;o.textContent=m.shard?`${m.name} [${m.start}..${m.stop-1}]`:m.name;sel.appendChild(o)}if([...sel.options].some(o=>o.value===old))sel.value=old;ensureMap(sel.value)}
async function tick(){try{let r=await fetch('/api/status',{cache:'no-store'});status=await r.json();document.querySelector('#link').textContent=status.nats?'NATS подключён':'NATS отключён';document.querySelector('#link').className='badge '+(status.nats?'ok':'bad');
 let age=status.age_s;document.querySelector('#fresh').textContent=age==null?'нет телеметрии':`телеметрия ${age.toFixed(1)}с`;document.querySelector('#fresh').className='badge '+(age!=null&&age<2?'ok':'bad');
 let tl=document.querySelector('#traffic');tl.textContent=status.traffic_loading?'Светофор: загрузка':`Светофор ${status.traffic_enabled?'ON':'OFF'}${status.traffic_state?' '+status.traffic_state:''}`;tl.className='badge '+(status.traffic_enabled?'ok':'');
 let recovery=document.querySelector('#recovery');recovery.textContent=status.full_map_recovery?'FULL RECOVERY':'SHARD';recovery.className='badge '+(status.full_map_recovery?'bad':'ok');
 document.querySelector('#mode').textContent=(status.mode||'—')+' '+(status.cam||'');if(status.route&&document.activeElement!==document.querySelector('#routeSelect'))document.querySelector('#routeSelect').value=status.route;document.querySelector('#command').textContent=(status.move_type||'LOST').toUpperCase()+(status.deg!=null?` ${status.deg>0?'+':''}${status.deg}°`:'');
 document.querySelector('#node').textContent=status.node==null?'—':status.node;document.querySelector('#offset').textContent=status.dist_to_route_m!=null?status.dist_to_route_m+' м':(status.dist_to_route??'—');document.querySelector('#inliers').textContent=status.inliers??'—';draw()}catch(e){document.querySelector('#message').textContent=e}setTimeout(tick,250)}
document.querySelectorAll('button').forEach(b=>b.onclick=async()=>{let body={cmd:b.dataset.cmd};if(b.dataset.mode)body.mode=b.dataset.mode;if(b.dataset.enabled)body.enabled=b.dataset.enabled==='true';let r=await fetch('/api/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});document.querySelector('#message').textContent=(await r.json()).message});
document.querySelector('#camera').onchange=fillMaps;document.querySelector('#mapSelect').onchange=e=>ensureMap(e.target.value);
document.querySelector('#setMap').onclick=async()=>{let body={cmd:'set_map',camera:document.querySelector('#camera').value,map:document.querySelector('#mapSelect').value};let r=await fetch('/api/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});document.querySelector('#message').textContent=(await r.json()).message};
document.querySelector('#setRoute').onclick=async()=>{let body={cmd:'set_route',route:document.querySelector('#routeSelect').value};let r=await fetch('/api/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});document.querySelector('#message').textContent=(await r.json()).message};
fetch('/api/maps').then(r=>r.json()).then(x=>{catalog=x.maps;fillMaps()});tick();
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


def map_catalog() -> list[dict[str, object]]:
    result = []
    for path in sorted(config.MAPS_DIR.iterdir()):
        if not path.is_dir() or not (path / "runtime.npz").exists() or not (path / "aliked_bank.npz").exists():
            continue
        camera = "front" if "front" in path.name else "rear" if "rear" in path.name else "unknown"
        item: dict[str, object] = {"name": path.name, "camera": camera, "shard": False}
        metadata = path / "shard.json"
        if metadata.exists():
            shard = json.loads(metadata.read_text())
            item.update({"shard": True, "number": shard["number"], "parts": shard["parts"],
                         "start": shard["global_node_start"], "stop": shard["global_node_stop_exclusive"],
                         "core_start": shard["core_global_start"],
                         "core_stop": shard["core_global_stop_exclusive"]})
        result.append(item)
    return result


async def main() -> int:
    parser = argparse.ArgumentParser(description="веб-админка локализации")
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--nats-url", default=config.NATS_URL)
    parser.add_argument("--rear-map", default=config.REAR_MAP)
    parser.add_argument("--front-map", default=config.FRONT_MAP)
    args = parser.parse_args()

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
                await respond(writer, "200 OK", json.dumps({"maps": catalog}).encode(), "application/json")
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
                allowed = {"pause", "resume", "reset", "set_mode", "set_map", "set_route", "set_traffic"}
                if command.get("cmd") not in allowed:
                    await respond(writer, "400 Bad Request", b'{"message":"unknown command"}', "application/json")
                    return
                if (command.get("cmd") == "set_map"
                        and (command.get("camera") not in ("front", "rear")
                             or command.get("map") not in allowed_maps)):
                    await respond(writer, "400 Bad Request", b'{"message":"invalid map"}', "application/json")
                    return
                if (command.get("cmd") == "set_route"
                        and command.get("route") not in config.ROUTES):
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
