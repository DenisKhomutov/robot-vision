import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(ROOT))
from localizer import AlikedLocalizer  # noqa: E402
import config as cfg  # noqa: E402


def xz_to_latlon(geo, x, z):
    (a, b, c), (d, e, f) = geo["affine"]
    east = a * x + b * z + c
    north = d * x + e * z + f
    lat = geo["lat0"] + north / geo["lat_m"]
    lon = geo["lon0"] + east / (math.cos(math.radians(geo["lat0"])) * geo["lon_m"])
    return lat, lon


HTML = """<!doctype html>
<meta charset="utf-8">
<title>geotrack — {map}</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<style>
  html,body{{margin:0;height:100%;font-family:system-ui,Arial,sans-serif;background:#111;color:#eee}}
  #wrap{{display:flex;flex-direction:column;height:100%}}
  header{{padding:8px 14px;background:#1a1a1a;border-bottom:1px solid #333;font-size:14px}}
  header b{{color:#2b8cff}} #st{{float:right;color:#8c8}}
  #panes{{flex:1;display:flex;min-height:0}}
  #left{{flex:1;display:flex;align-items:center;justify-content:center;background:#000;min-width:0}}
  #left video{{max-width:100%;max-height:100%}}
  #map{{flex:1;min-width:0}}
  @media(max-width:800px){{#panes{{flex-direction:column}}}}
</style>
<div id="wrap">
  <header>Маршрут <b>{map}</b> — видео и РЕАЛЬНАЯ позиция локализатора
    <span id="st">фиксов {nfix}/{ntot}</span></header>
  <div id="panes">
    <div id="left"><video id="vid" src="{src}" controls autoplay muted loop playsinline></video></div>
    <div id="map"></div>
  </div>
</div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const ROUTE = {route};      // эталон (для фона)
const TRACK = {track};      // [[t_sec, lat, lon], ...] РЕАЛЬНЫЕ фиксы, по времени видео
const map = L.map("map");
L.tileLayer("https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png",
  {{maxZoom: 20, attribution: "© OpenStreetMap"}}).addTo(map);
const line = L.polyline(ROUTE, {{color: "#2b8cff", weight: 3, opacity: 0.5}}).addTo(map);
map.fitBounds(line.getBounds().pad(0.15));
const robot = L.circleMarker(TRACK.length ? [TRACK[0][1], TRACK[0][2]] : ROUTE[0],
  {{radius: 9, color: "#fff", weight: 2, fillColor: "#e23", fillOpacity: 1}}).addTo(map);

const video = document.getElementById("vid");
// позиция по времени видео: бинарный поиск в треке + интерполяция
function posAt(tc) {{
  if (!TRACK.length) return null;
  let lo = 0, hi = TRACK.length - 1;
  if (tc <= TRACK[0][0]) return [TRACK[0][1], TRACK[0][2]];
  if (tc >= TRACK[hi][0]) return [TRACK[hi][1], TRACK[hi][2]];
  while (hi - lo > 1) {{ const m = (lo + hi) >> 1; (TRACK[m][0] <= tc ? lo = m : hi = m); }}
  const a = TRACK[lo], b = TRACK[hi], f = (tc - a[0]) / Math.max(b[0] - a[0], 1e-6);
  return [a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}}
let cur = TRACK.length ? [TRACK[0][1], TRACK[0][2]] : null;
function step() {{
  const tgt = posAt(video.currentTime);
  if (tgt && cur) {{                                   // лёгкое сглаживание, чтобы не дёргалось
    cur = [cur[0] + (tgt[0] - cur[0]) * 0.2, cur[1] + (tgt[1] - cur[1]) * 0.2];
    robot.setLatLng(cur);
  }}
  requestAnimationFrame(step);
}}
step();
</script>
"""


def main():
    ap = argparse.ArgumentParser(description="Прогон видео по карте -> реальный гео-трек + страница")
    ap.add_argument("--map", default="map_street_1_clear")
    ap.add_argument("--video", required=True)
    ap.add_argument("--step", type=int, default=3, help="локализовать каждый N-й кадр видео")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    work = ROOT / "maps" / args.map
    geo = json.loads((work / "geo.json").read_text())
    loc = AlikedLocalizer(args.map, kpts=cfg.QUERY_KPTS, det_threshold=cfg.QUERY_DET_THRESHOLD,
                          nms_radius=cfg.QUERY_NMS_RADIUS, max_error=cfg.MAX_ERROR,
                          match_ratio=cfg.MATCH_RATIO, match_topk=cfg.MATCH_TOPK,
                          focal_fallback=cfg.FOCAL_FALLBACK, min_pairs=cfg.MIN_PAIRS)
    m = np.load(work / "runtime.npz")
    pos = m["pos"]
    route = [list(xz_to_latlon(geo, pos[i, 0], pos[i, 2])) for i in range(0, len(pos), 2)]

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"не открывается видео: {args.video}")
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    track = []
    idx = ntot = 0
    t0 = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % args.step == 0:
            ntot += 1
            r = loc.locate(frame)
            if r.get("ok") and r.get("inliers", 0) >= cfg.MIN_INLIERS:
                C = r["C"]
                lat, lon = xz_to_latlon(geo, float(C[0]), float(C[2]))
                track.append([round(idx / fps, 3), round(lat, 7), round(lon, 7)])
            if ntot % 50 == 0:
                el = time.perf_counter() - t0
                print(f"  {idx}/{total} кадр | фиксов {len(track)}/{ntot} | {el:.0f}с", flush=True)
        idx += 1
    cap.release()

    out = Path(args.out) if args.out else ROOT / "out" / f"geotrack_{args.map}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    vp = Path(args.video).resolve()
    src = os.path.relpath(vp, out.parent) if vp.exists() else args.video
    out.write_text(HTML.format(map=args.map, src=src, nfix=len(track), ntot=ntot,
                               route=json.dumps(route), track=json.dumps(track)))
    print(f"\nфиксов {len(track)}/{ntot} ({100*len(track)/max(ntot,1):.0f}%) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())