from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from .. import config
from ..core.hub import use_local_weights
from .build_compact_bank import artifact_dir, import_faiss, normalize


class CompactFullMapLocalizer:
    def __init__(self, map_name: str, bank_path: Path, backend: str,
                 index_path: Path | None, nprobe: int, search_k: int):
        from lightglue import ALIKED

        use_local_weights()
        self.backend = backend
        self.search_k = search_k
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        work = config.MAPS_DIR / map_name
        runtime = np.load(work / "runtime.npz")
        self.names = runtime["names"]
        self.route = runtime["pos"].astype(np.float64)
        self.K = np.array([
            [runtime["K"][0], 0, runtime["K"][2]],
            [0, runtime["K"][1], runtime["K"][3]],
            [0, 0, 1],
        ], dtype=np.float64)
        self.dist = runtime["dist"].astype(np.float64)
        self.map_size = tuple(int(value) for value in runtime["size"])
        bank = np.load(bank_path)
        self.desc_np = normalize(bank["desc"])
        self.owner_np = bank["owner"].astype(np.int64, copy=False)
        self.xyz = bank["xyz"].astype(np.float64, copy=False)
        self.extractor = ALIKED(
            max_num_keypoints=config.QUERY_KPTS,
            detection_threshold=config.QUERY_DET_THRESHOLD,
            nms_radius=config.QUERY_NMS_RADIUS,
        ).eval().to(self.device)
        self.half = self.device == "cuda"
        if self.half:
            self.extractor = self.extractor.half()
        self.desc_torch = None
        self.owner_torch = None
        self.index = None
        if backend == "torch-exact":
            dtype = torch.float16 if self.half else torch.float32
            self.desc_torch = torch.from_numpy(self.desc_np).to(self.device, dtype=dtype)
            self.owner_torch = torch.from_numpy(self.owner_np).to(self.device)
        else:
            faiss = import_faiss()
            if index_path is None or not index_path.exists():
                raise FileNotFoundError(f"нет FAISS-индекса: {index_path}")
            self.index = faiss.read_index(str(index_path))
            if hasattr(self.index, "nprobe"):
                self.index.nprobe = nprobe

    def extract(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        started = time.perf_counter()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = torch.from_numpy(rgb.transpose(2, 0, 1)).to(self.device)
        if self.half:
            image = image.half()
        with torch.inference_mode():
            features = self.extractor.extract(image)
        keypoints = features["keypoints"][0].cpu().numpy()
        descriptors = features["descriptors"][0]
        elapsed = (time.perf_counter() - started) * 1000
        return keypoints, descriptors, elapsed

    def torch_search(self, query: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        query = query.to(self.device, dtype=self.desc_torch.dtype)
        count = len(query)
        k = min(self.search_k, len(self.owner_np))
        best_scores = torch.empty((count, 0), dtype=query.dtype, device=self.device)
        best_ids = torch.empty((count, 0), dtype=torch.long, device=self.device)
        chunk = max(4096, int(128e6 / (2 * max(count, 1))))
        with torch.inference_mode():
            for start in range(0, len(self.owner_np), chunk):
                similarity = query @ self.desc_torch[start:start + chunk].T
                local_k = min(k, similarity.shape[1])
                values, indices = similarity.topk(local_k, dim=1)
                indices += start
                values = torch.cat((best_scores, values), dim=1)
                indices = torch.cat((best_ids, indices), dim=1)
                best_scores, selected = values.topk(min(k, values.shape[1]), dim=1)
                best_ids = indices.gather(1, selected)
        return best_scores.float().cpu().numpy(), best_ids.cpu().numpy()

    def search(self, query: torch.Tensor) -> tuple[np.ndarray, np.ndarray, float]:
        started = time.perf_counter()
        if self.backend == "torch-exact":
            scores, ids = self.torch_search(query)
        else:
            query_np = normalize(query.float().cpu().numpy())
            scores, ids = self.index.search(np.ascontiguousarray(query_np), self.search_k)
        return scores, ids, (time.perf_counter() - started) * 1000

    def correspondences(self, keypoints: np.ndarray, scores: np.ndarray,
                        ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        candidates = []
        for query_id, (row_scores, row_ids) in enumerate(zip(scores, ids)):
            distinct = []
            seen = set()
            for score, prototype_id in zip(row_scores, row_ids):
                if prototype_id < 0:
                    continue
                point_id = int(self.owner_np[int(prototype_id)])
                if point_id in seen:
                    continue
                seen.add(point_id)
                distinct.append((float(score), point_id))
                if len(distinct) == 2:
                    break
            if len(distinct) < 2:
                continue
            d1 = np.sqrt(max(0.0, 2.0 - 2.0 * distinct[0][0]))
            d2 = np.sqrt(max(1e-12, 2.0 - 2.0 * distinct[1][0]))
            if d1 / d2 < config.MATCH_RATIO:
                candidates.append((distinct[0][0], query_id, distinct[0][1]))
        candidates.sort(reverse=True)
        selected = []
        used_points = set()
        for _, query_id, point_id in candidates:
            if point_id in used_points:
                continue
            used_points.add(point_id)
            selected.append((query_id, point_id))
        if not selected:
            return np.empty((0, 2)), np.empty((0, 3))
        return (
            np.asarray([keypoints[query_id] for query_id, _ in selected]),
            np.asarray([self.xyz[point_id] for _, point_id in selected]),
        )

    def locate(self, frame: np.ndarray) -> dict[str, object]:
        keypoints, query, extract_ms = self.extract(frame)
        scores, ids, match_ms = self.search(query)
        points2d, points3d = self.correspondences(keypoints, scores, ids)
        result = {
            "ok": False,
            "pairs": int(len(points2d)),
            "inliers": 0,
            "extract_ms": extract_ms,
            "match_ms": match_ms,
            "pnp_ms": 0.0,
        }
        if len(points2d) < config.MIN_PAIRS:
            result["reason"] = "мало пар"
            return result
        height, width = frame.shape[:2]
        K, dist = self.K, self.dist
        if (width, height) != self.map_size:
            K = K.copy()
            K[0] *= width / self.map_size[0]
            K[1] *= height / self.map_size[1]
        started = time.perf_counter()
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            points3d, points2d.astype(np.float64), K, dist,
            reprojectionError=config.MAX_ERROR,
            confidence=0.999,
            iterationsCount=1000,
            flags=cv2.SOLVEPNP_EPNP,
        )
        result["pnp_ms"] = (time.perf_counter() - started) * 1000
        result["inliers"] = 0 if inliers is None else int(len(inliers))
        if not ok or inliers is None or len(inliers) < config.MIN_INLIERS:
            result["reason"] = "PnP не сошёлся или мало inliers"
            return result
        selected = inliers.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(
            points3d[selected], points2d[selected].astype(np.float64), K, dist, rvec, tvec,
        )
        projected, _ = cv2.projectPoints(points3d[selected], rvec, tvec, K, dist)
        residual = np.linalg.norm(projected.reshape(-1, 2) - points2d[selected], axis=1)
        result["reproj_error"] = float(np.sqrt(np.mean(residual ** 2)))
        result["pnp_ms"] = (time.perf_counter() - started) * 1000
        rotation, _ = cv2.Rodrigues(rvec)
        center = (-rotation.T @ tvec).ravel()
        node = int(np.argmin(np.linalg.norm(self.route - center, axis=1)))
        result.update({"ok": True, "node": node, "C": center.tolist()})
        return result


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values), q))


def main() -> int:
    parser = argparse.ArgumentParser(description="Прогон видео по единой компактной карте")
    parser.add_argument("video", type=Path)
    parser.add_argument("--map", required=True)
    parser.add_argument("--artifact", type=Path, default=None)
    parser.add_argument("--bank", choices=("compact", "original"), default="compact")
    parser.add_argument("--backend", choices=("torch-exact", "faiss-flat", "faiss-ivf"), default="torch-exact")
    parser.add_argument("--nprobe", type=int, default=32)
    parser.add_argument("--search-k", type=int, default=16)
    parser.add_argument("--step", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    artifact = args.artifact or artifact_dir(args.map)
    if args.bank == "original":
        bank_path = config.MAPS_DIR / args.map / "aliked_bank.npz"
    else:
        bank_path = artifact / "compact_bank.npz"
    index_path = None
    if args.backend == "faiss-flat":
        index_path = artifact / "compact_flat.faiss"
    elif args.backend == "faiss-ivf":
        index_path = artifact / "compact_ivf.faiss"
    if args.bank == "original" and args.backend != "torch-exact":
        raise SystemExit("FAISS-индексы создаются для компактного банка; используйте --bank compact")

    args.out.mkdir(parents=True, exist_ok=True)
    diagnostics_path = args.out / "diagnostics.jsonl"
    summary_path = args.out / "summary.json"
    video_path = args.out / "result.mp4"
    localizer = CompactFullMapLocalizer(
        args.map, bank_path, args.backend, index_path, args.nprobe, args.search_k,
    )
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"не открывается видео: {args.video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, fps / args.step), (width, height),
    )
    rows = []
    frame_id = 0
    processed = 0
    started_all = time.perf_counter()
    with diagnostics_path.open("w", encoding="utf-8") as diagnostics:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_id % args.step:
                frame_id += 1
                continue
            result = localizer.locate(frame)
            total_ms = result["extract_ms"] + result["match_ms"] + result["pnp_ms"]
            row = {"source_frame": frame_id, "total_ms": total_ms, **result}
            diagnostics.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            rows.append(row)
            label = (
                f"node={result.get('node')} inliers={result['inliers']} {total_ms:.0f}ms"
                if result["ok"] else f"LOST {result.get('reason')} {total_ms:.0f}ms"
            )
            color = (70, 220, 90) if result["ok"] else (40, 60, 240)
            cv2.putText(frame, label, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            writer.write(frame)
            processed += 1
            if processed % 25 == 0:
                print(f"[{processed}] frame={frame_id} {label}", flush=True)
            frame_id += 1
            if args.max_frames is not None and processed >= args.max_frames:
                break
    capture.release()
    writer.release()
    elapsed = time.perf_counter() - started_all
    totals = [float(row["total_ms"]) for row in rows]
    matches = [float(row["match_ms"]) for row in rows]
    fixes = [row for row in rows if row["ok"]]
    jumps = [
        abs(int(right["node"]) - int(left["node"]))
        for left, right in zip(fixes, fixes[1:])
    ]
    summary = {
        "video": str(args.video),
        "map": args.map,
        "bank": args.bank,
        "bank_path": str(bank_path),
        "backend": args.backend,
        "nprobe": args.nprobe if args.backend == "faiss-ivf" else None,
        "processed_frames": processed,
        "fixes": len(fixes),
        "fix_rate": 0.0 if not rows else len(fixes) / len(rows),
        "total_p50_ms": percentile(totals, 50),
        "total_p95_ms": percentile(totals, 95),
        "match_p50_ms": percentile(matches, 50),
        "match_p95_ms": percentile(matches, 95),
        "max_node_jump": max(jumps) if jumps else None,
        "wall_seconds": elapsed,
        "device": localizer.device,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[готово] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
