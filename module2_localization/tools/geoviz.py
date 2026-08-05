import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def xz_to_latlon(geo, x, z):
    (a, b, c), (d, e, f) = geo["affine"]
    east = a * x + b * z + c
    north = d * x + e * z + f
    lat = geo["lat0"] + north / geo["lat_m"]
    lon = geo["lon0"] + east / (math.cos(math.radians(geo["lat0"])) * geo["lon_m"])
    return lat, lon


HTML = """<!doctype html>
<meta charset="utf-8">
<title>geoviz — {map}</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<style>
  html,body{{margin:0;height:100%;font-family:system-ui,Arial,sans-serif;background:#111;color:#eee}}
  #wrap{{display:flex;flex-direction:column;height:100%}}
  header{{padding:8px 14px;background:#1a1a1a;border-bottom:1px solid #333;font-size:14px}}
  header b{{color:#2b8cff}}
  #panes{{flex:1;display:flex;min-height:0}}
  #left{{flex:1;display:flex;align-items:center;justify-content:center;background:#000;min-width:0}}
  #left video{{max-width:100%;max-height:100%}}
  #map{{flex:1;min-width:0}}
  @media(max-width:800px){{#panes{{flex-direction:column}}}}
</style>
<div id="wrap">
  <header>Маршрут <b>{map}</b> — видео и позиция на карте (привязка: {method}, СКО ~{rms} м)</header>
  <div id="panes">
    <div id="left">{video_pane}</div>
    <div id="map"></div>
  </div>
</div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const ROUTE = {route};
const CTRL  = {ctrl};
const map = L.map("map");
L.tileLayer("https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png",
  {{maxZoom: 20, attribution: "© OpenStreetMap"}}).addTo(map);
const line = L.polyline(ROUTE, {{color: "#2b8cff", weight: 4, opacity: 0.85}}).addTo(map);
map.fitBounds(line.getBounds().pad(0.15));
CTRL.forEach((p,i) => L.circleMarker(p,
  {{radius: 5, color: "#888", fillColor: "#bbb", fillOpacity: 1}})
  .bindTooltip("ctrl "+i).addTo(map));
const robot = L.circleMarker(ROUTE[0],
  {{radius: 9, color: "#fff", weight: 2, fillColor: "#e23", fillOpacity: 1}}).addTo(map);

let targetIdx = 0;                 // куда хотим (задаётся видео или авто-анимацией)
const video = document.getElementById("vid");
if (video) {{
  video.addEventListener("timeupdate", () => {{
    if (video.duration) targetIdx = (video.currentTime / video.duration) * (ROUTE.length - 1);
  }});
}} else {{
  let a = 0; setInterval(() => {{ a = (a + 0.6) % (ROUTE.length - 1); targetIdx = a; }}, 40);
}}

let cur = 0;                       // текущий индекс маркера, плавно догоняет target
function step() {{
  cur += (targetIdx - cur) * 0.15;                 // сглаживание (инерция)
  const i = Math.max(0, Math.min(ROUTE.length - 2, Math.floor(cur)));
  const f = cur - i, a = ROUTE[i], b = ROUTE[i + 1];
  robot.setLatLng([a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f]);
  requestAnimationFrame(step);
}}
step();
</script>
"""

VIDEO_PANE = '<video id="vid" src="{src}" controls autoplay muted loop playsinline></video>'
NO_VIDEO = '<div style="color:#666">видео не задано (--video путь.mp4)</div>'


def main():
    ap = argparse.ArgumentParser(description="Геопривязка: видео маршрута + позиция на карте")
    ap.add_argument("--map", default="map_street_1_clear")
    ap.add_argument("--video", default=None, help="видео маршрута рядом с картой")
    ap.add_argument("--step", type=int, default=2, help="брать каждый N-й узел")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    work = ROOT / "maps" / args.map
    geo = json.loads((work / "geo.json").read_text())
    m = np.load(work / "runtime.npz")
    pos = m["pos"]
    names = [str(x) for x in m["names"]]

    route = [list(xz_to_latlon(geo, pos[i, 0], pos[i, 2])) for i in range(0, len(pos), args.step)]
    ctrl = []
    for fr in geo.get("control_frames", []):
        j = next((k for k, n in enumerate(names) if f"{fr:05d}" in n), None)
        if j is not None:
            ctrl.append(list(xz_to_latlon(geo, pos[j, 0], pos[j, 2])))

    out = Path(args.out) if args.out else ROOT / "out" / f"geoviz_{args.map}.html"
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.video:
        vp = Path(args.video).resolve()
        src = os.path.relpath(vp, out.parent) if vp.exists() else args.video
        video_pane = VIDEO_PANE.format(src=src)
    else:
        video_pane = NO_VIDEO

    out.write_text(HTML.format(
        map=args.map, method=geo.get("method", "?"), rms=round(geo.get("rms_m", 0), 1),
        route=json.dumps(route), ctrl=json.dumps(ctrl), video_pane=video_pane))
    print(f"узлов {len(route)}, контрольных {len(ctrl)} -> {out}")
    print("открой в браузере" + ("" if args.video else "  (видео: добавь --video путь.mp4)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())