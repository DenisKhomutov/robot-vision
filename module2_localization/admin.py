"""Browser dashboard for localization status and NATS controls on Jetson."""



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
<title>Robot Vision</title>
<style>
:root{
  color-scheme:dark;font-family:system-ui,-apple-system,sans-serif;
  --bg:#0d0f13;--panel:#161a21;--panel2:#1d222b;--border:#2a3140;
  --text:#eef0f3;--muted:#8b95a5;
  --green:#1e8a53;--green-bg:#123524;--red:#c1453a;--red-bg:#3a1a17;
  --yellow:#e0ac3d;--yellow-bg:#3a2c10;--accent:#4c7cf0;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);line-height:1.4}
header{
  display:flex;gap:10px;align-items:center;padding:12px 20px;background:var(--panel);
  border-bottom:1px solid var(--border);flex-wrap:wrap;position:sticky;top:0;z-index:10
}
header h1{font-size:1.05rem;margin:0 14px 0 0;font-weight:700;letter-spacing:.02em}
.badge{
  display:inline-flex;align-items:center;gap:6px;padding:5px 11px;border-radius:20px;
  background:var(--panel2);font-size:.82rem;color:var(--muted);border:1px solid var(--border)
}
.badge::before{content:'';width:7px;height:7px;border-radius:50%;background:var(--muted)}
.badge.ok{background:var(--green-bg);color:#a9e6c4;border-color:#1e5c3c}
.badge.ok::before{background:var(--green)}
.badge.bad{background:var(--red-bg);color:#f3b6ae;border-color:#6b2c25}
.badge.bad::before{background:var(--red)}
.badge.warn{background:var(--yellow-bg);color:#f5d99b;border-color:#7a5c1f}
.badge.warn::before{background:var(--yellow)}

main{display:grid;grid-template-columns:minmax(0,1fr) 360px;gap:14px;padding:14px;max-width:1400px;margin:auto}
.map-wrap{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:10px}
canvas{display:block;width:100%;height:auto;background:#0a0c10;border-radius:8px}

aside{display:flex;flex-direction:column;gap:12px}
.card{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:14px}
.card h2{font-size:.72rem;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);margin:0 0 10px;font-weight:600}

.status-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px 14px;margin-bottom:2px}
.stat label{display:block;font-size:.74rem;color:var(--muted);margin-bottom:2px}
.stat .value{font-size:1.3rem;font-weight:700}
.stat.command .value{font-size:1.55rem}
.stat.command .value.go{color:#5fd694}
.stat.command .value.stop{color:#ff8a7a}
.stat.command .value.lost{color:var(--yellow)}

.row2{display:grid;grid-template-columns:1fr 1fr;gap:8px}
button{
  font-size:.95rem;padding:12px;border:1px solid var(--border);border-radius:9px;
  background:var(--panel2);color:var(--text);cursor:pointer;transition:filter .1s,transform .05s
}
button:hover{filter:brightness(1.15)}
button:active{transform:scale(.97)}
button.big{padding:16px;font-size:1.05rem;font-weight:700}
button.go{background:var(--green-bg);border-color:#1e5c3c;color:#bdf0d3}
button.stop{background:var(--red-bg);border-color:#6b2c25;color:#f6c3bb}
button.toggle{position:relative}
button.toggle.active{background:var(--accent);border-color:var(--accent);color:#fff;font-weight:700}
button.ghost{background:transparent}
button:disabled,select:disabled{opacity:.4;cursor:not-allowed;filter:none}
button:disabled:hover{filter:none}

select{
  width:100%;padding:10px;border-radius:8px;border:1px solid var(--border);
  background:var(--panel2);color:var(--text);font-size:.9rem;margin-bottom:8px
}
label.field{display:block;font-size:.78rem;color:var(--muted);margin:10px 0 4px}

#message{min-height:1.4em;color:var(--yellow);font-size:.85rem;margin-top:8px}
.legend{display:flex;gap:14px;font-size:.75rem;color:var(--muted);margin-top:8px;flex-wrap:wrap}
.legend span{display:inline-flex;align-items:center;gap:5px}
.legend i{width:10px;height:10px;border-radius:50%;display:inline-block}

@media(max-width:900px){main{grid-template-columns:1fr}aside{order:-1}}
</style></head>
<body>
<header>
  <h1>ROBOT VISION</h1>
  <span id="link" class="badge bad">NATS</span>
  <span id="fresh" class="badge bad">нет телеметрии</span>
  <span id="mode" class="badge">—</span>
  <span id="traffic" class="badge">светофор off</span>
  <span id="direction" class="badge">направление off</span>
  <span id="recovery" class="badge ok">shard</span>
</header>
<main>
  <div class="map-wrap">
    <div class="row2" style="grid-template-columns:1fr 1fr;margin-bottom:10px">
      <button class="toggle active" id="tabShard" data-tab="shard">ШАРД</button>
      <button class="toggle" id="tabFull" data-tab="full">ЕДИНАЯ КАРТА</button>
    </div>
    <canvas id="map" width="900" height="760"></canvas>
    <div class="legend">
      <span><i style="background:#d4ae45"></i>маршрут</span>
      <span><i style="background:#ff4b4b"></i>текущая позиция</span>
      <span><i style="background:#555"></i>облако точек карты</span>
    </div>
  </div>
  <aside>
    <section class="card">
      <h2>Состояние</h2>
      <div class="status-grid">
        <div class="stat command"><label>Команда</label><div id="command" class="value lost">LOST</div></div>
        <div class="stat"><label>Узел</label><div id="node" class="value">—</div></div>
        <div class="stat"><label>Отклонение</label><div id="offset" class="value">—</div></div>
        <div class="stat"><label>Инлайнеры</label><div id="inliers" class="value">—</div></div>
      </div>
    </section>

    <section class="card">
      <h2>Навигация</h2>
      <div class="row2">
        <button class="go big toggle" id="btnResume" data-cmd="resume">СТАРТ</button>
        <button class="stop big toggle" id="btnPause" data-cmd="pause">ПАУЗА</button>
      </div>
      <button class="ghost" data-cmd="reset" style="width:100%;margin-top:8px">СБРОС</button>
      <button class="ghost" data-cmd="reset_shard" style="width:100%;margin-top:8px">СБРОС ШАРДА (релокализация)</button>
    </section>

    <section class="card">
      <h2>Режим камер</h2>
      <div class="row2" style="grid-template-columns:1fr 1fr 1fr">
        <button class="toggle gated" id="btnFront" data-cmd="set_mode" data-mode="front">FRONT</button>
        <button class="toggle gated" id="btnRear" data-cmd="set_mode" data-mode="rear">REAR</button>
        <button class="toggle gated" id="btnDual" data-cmd="set_mode" data-mode="dual">DUAL</button>
      </div>
    </section>

    <section class="card">
      <h2>Светофор</h2>
      <div class="row2">
        <button class="toggle" id="btnTlOn" data-cmd="set_traffic" data-enabled="true">ВКЛ</button>
        <button class="toggle" id="btnTlOff" data-cmd="set_traffic" data-enabled="false">ВЫКЛ</button>
      </div>
      <button class="ghost" data-cmd="reset_traffic" style="width:100%;margin-top:8px">СБРОС СВЕТОФОРА</button>
    </section>

    <section class="card">
      <h2>Направление (эксперимент)</h2>
      <div class="row2">
        <button class="toggle" id="btnDirOn" data-cmd="set_direction" data-enabled="true">ВКЛ</button>
        <button class="toggle" id="btnDirOff" data-cmd="set_direction" data-enabled="false">ВЫКЛ</button>
      </div>
    </section>

    <section class="card">
      <h2>Маршрут</h2>
      <select id="routeSelect" class="gated"></select>
      <button id="setRoute" class="gated" style="width:100%">ВЫБРАТЬ МАРШРУТ</button>
    </section>

    <p id="message"></p>
  </aside>
</main>
<script>
let maps={},routes={},status={},catalog=[],activeTab='shard';
const canvas=document.querySelector('#map'),ctx=canvas.getContext('2d');
const $=s=>document.querySelector(s);

function fullMapName(){
  let route=status.route;
  if(!route)return null;
  let hit=catalog.find(m=>!m.shard&&m.name.startsWith(route+'/'));
  return hit?hit.name:null;
}

function fit(points){
  let xs=points.map(p=>p[0]),ys=points.map(p=>p[1]);
  let lo=[Math.min(...xs),Math.min(...ys)],hi=[Math.max(...xs),Math.max(...ys)];
  return p=>[35+(p[0]-lo[0])/(hi[0]-lo[0]||1)*(canvas.width-70),
             canvas.height-35-(p[1]-lo[1])/(hi[1]-lo[1]||1)*(canvas.height-70)];
}

function draw(){
  let name=activeTab==='full'?fullMapName():status.map,m=name?maps[name]:null;
  if(!m){if(name)ensureMap(name);return}
  let all=m.cloud.length?m.cloud:m.route,px=fit(all);
  ctx.fillStyle='#0a0c10';ctx.fillRect(0,0,canvas.width,canvas.height);
  ctx.fillStyle='#4a5160';for(let p of m.cloud){let q=px(p);ctx.fillRect(q[0],q[1],1,1)}
  ctx.strokeStyle='#d4ae45';ctx.lineWidth=3;ctx.beginPath();
  m.route.forEach((p,i)=>{let q=px(p);i?ctx.lineTo(...q):ctx.moveTo(...q)});ctx.stroke();
  if(status.pos){
    let q=px(status.pos);
    ctx.fillStyle='#ff4b4b';ctx.beginPath();ctx.arc(q[0],q[1],8,0,7);ctx.fill();
    ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.stroke();
    if(status.head){
      let h=px([status.pos[0]+status.head[0]*.5,status.pos[1]+status.head[1]*.5]);
      ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(...q);ctx.lineTo(...h);ctx.stroke();
    }
  }
  ctx.fillStyle='#eee';ctx.font='600 16px system-ui';ctx.fillText(m.name,16,24);
}

async function ensureMap(name){
  if(!name||maps[name])return;
  let r=await fetch('/api/map?name='+encodeURIComponent(name));
  if(r.ok){maps[name]=await r.json();draw()}
}

function fillRoutes(){
  let sel=$('#routeSelect'),old=sel.value;
  sel.innerHTML='';
  let empty=document.createElement('option');
  empty.value='';empty.textContent='— маршрут не выбран —';
  sel.appendChild(empty);
  for(let key in routes){
    let o=document.createElement('option');
    o.value=key;o.textContent=routes[key];
    sel.appendChild(o);
  }
  if([...sel.options].some(o=>o.value===old))sel.value=old;
}

function setBadge(el,text,cls){el.textContent=text;el.className='badge '+cls}

async function tick(){
  try{
    let r=await fetch('/api/status',{cache:'no-store'});
    status=await r.json();

    setBadge($('#link'),status.nats?'NATS подключён':'NATS отключён',status.nats?'ok':'bad');

    let age=status.age_s;
    setBadge($('#fresh'),age==null?'нет телеметрии':`телеметрия ${age.toFixed(1)}с`,
      age!=null&&age<2?'ok':'bad');

    setBadge($('#traffic'),
      status.traffic_loading?'светофор: загрузка':`светофор ${status.traffic_enabled?'вкл':'выкл'}${status.traffic_state?' · '+status.traffic_state:''}`,
      status.traffic_loading?'warn':(status.traffic_enabled?'ok':''));

    setBadge($('#recovery'),status.full_map_recovery?'full recovery':'shard',
      status.full_map_recovery?'warn':'ok');

    setBadge($('#direction'),
      `направление ${status.direction_enabled?'вкл':'выкл'}${status.direction?' · '+status.direction:''}`,
      status.direction_enabled?'ok':'');

    $('#mode').textContent=(status.mode||'—')+(status.cam?' · '+status.cam:'');
    $('#mode').className='badge'+(status.map_mismatch?' bad':'');

    if(status.route&&document.activeElement!==$('#routeSelect'))$('#routeSelect').value=status.route;

    let mv=(status.move_type||'lost').toLowerCase();
    let cmdEl=$('#command');
    cmdEl.textContent=mv.toUpperCase()+(status.deg!=null?` ${status.deg>0?'+':''}${status.deg}°`:'');
    cmdEl.className='value '+(mv==='stop'?'stop':mv==='lost'?'lost':'go');

    $('#node').textContent=status.node==null?'—':status.node;
    $('#offset').textContent=status.dist_to_route_m!=null?status.dist_to_route_m+' м':(status.dist_to_route??'—');
    $('#inliers').textContent=status.inliers??'—';

    let paused=!!status.paused;
    $('#btnResume').classList.toggle('active',!paused);
    $('#btnPause').classList.toggle('active',paused);
    $('#btnFront').classList.toggle('active',status.mode==='front');
    $('#btnRear').classList.toggle('active',status.mode==='rear');
    $('#btnDual').classList.toggle('active',status.mode==='dual');
    $('#btnTlOn').classList.toggle('active',!!status.traffic_enabled);
    $('#btnTlOff').classList.toggle('active',!status.traffic_enabled);
    $('#btnDirOn').classList.toggle('active',!!status.direction_enabled);
    $('#btnDirOff').classList.toggle('active',!status.direction_enabled);

    document.querySelectorAll('.gated').forEach(el=>el.disabled=!paused);

    draw();
  }catch(e){$('#message').textContent=e}
  setTimeout(tick,250);
}

async function send(body){
  let r=await fetch('/api/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  $('#message').textContent=(await r.json()).message;
}

document.querySelectorAll('button[data-cmd]').forEach(b=>b.onclick=()=>{
  let body={cmd:b.dataset.cmd};
  if(b.dataset.mode)body.mode=b.dataset.mode;
  if(b.dataset.enabled)body.enabled=b.dataset.enabled==='true';
  send(body);
});
$('#setRoute').onclick=()=>send({cmd:'set_route',route:$('#routeSelect').value});
document.querySelectorAll('button[data-tab]').forEach(b=>b.onclick=()=>{
  activeTab=b.dataset.tab;
  $('#tabShard').classList.toggle('active',activeTab==='shard');
  $('#tabFull').classList.toggle('active',activeTab==='full');
  draw();
});

fetch('/api/routes').then(r=>r.json()).then(x=>{routes=x.routes;fillRoutes()});
fetch('/api/maps').then(r=>r.json()).then(x=>{catalog=x.maps});
tick();
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


    for runtime_file in sorted(config.MAPS_DIR.rglob("runtime.npz")):
        path = runtime_file.parent
        if not (path / "aliked_bank.npz").exists():
            continue
        name = path.relative_to(config.MAPS_DIR).as_posix()
        camera = "front" if "front" in name else "rear" if "rear" in name else "unknown"
        item: dict[str, object] = {"name": name, "camera": camera, "shard": False}
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
            elif method == "GET" and path == "/api/routes":
                routes = {key: str(spec.get("label", key)) for key, spec in config.ROUTES.items()}
                await respond(writer, "200 OK", json.dumps({"routes": routes}, ensure_ascii=False).encode(),
                              "application/json; charset=utf-8")
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
                allowed = {"pause", "resume", "reset", "reset_shard", "reset_traffic", "set_mode", "set_route", "set_traffic", "set_direction"}
                if command.get("cmd") not in allowed:
                    await respond(writer, "400 Bad Request", b'{"message":"unknown command"}', "application/json")
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
