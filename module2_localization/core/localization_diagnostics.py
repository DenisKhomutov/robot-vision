import cv2
import numpy as np


def distribution_stats(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    return {
        "min": float(values.min()),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "max": float(values.max()),
        "std": float(values.std()),
    }


def point_coverage(points, width, height, grid_cols=4, grid_rows=3):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if not len(points) or width <= 0 or height <= 0:
        return None
    x = np.clip(points[:, 0] / width, 0.0, 1.0)
    y = np.clip(points[:, 1] / height, 0.0, 1.0)
    cols = np.minimum((x * grid_cols).astype(np.int64), grid_cols - 1)
    rows = np.minimum((y * grid_rows).astype(np.int64), grid_rows - 1)
    occupied = np.unique(rows * grid_cols + cols)
    x0, x1 = float(x.min()), float(x.max())
    y0, y1 = float(y.min()), float(y.max())
    centered = np.column_stack((x - x.mean(), y - y.mean()))
    covariance = centered.T @ centered / max(len(points), 1)
    eigenvalues = np.linalg.eigvalsh(covariance)
    return {
        "count": int(len(points)),
        "grid": [grid_cols, grid_rows],
        "occupied_cells": int(len(occupied)),
        "occupied_fraction": float(len(occupied) / (grid_cols * grid_rows)),
        "bbox_norm": [x0, y0, x1, y1],
        "bbox_area_fraction": float((x1 - x0) * (y1 - y0)),
        "centroid_norm": [float(x.mean()), float(y.mean())],
        "spread_eigenvalues": [float(v) for v in eigenvalues],
    }


def points3d_geometry(points):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if not len(points):
        return None
    centered = points - points.mean(axis=0)
    covariance = centered.T @ centered / max(len(points), 1)
    eigenvalues = np.linalg.eigvalsh(covariance)
    return {
        "count": int(len(points)),
        "bbox_min": points.min(axis=0).tolist(),
        "bbox_max": points.max(axis=0).tolist(),
        "centroid": points.mean(axis=0).tolist(),
        "spread_eigenvalues": [float(v) for v in eigenvalues],
    }


def matching_diagnostics(
    query,
    query_count,
    bank_total,
    bank_searched,
    windowed_bank,
    previous_node,
    match_ratio,
    points2d,
    points3d,
    owners,
    best_distance,
    second_distance,
    ratios,
    keep,
):
    unique_owners = np.unique(owners)
    return {
        "query_token": int(query.get("query_token", 0)),
        "query_keypoints": int(query_count),
        "bank_descriptors_total": int(bank_total),
        "bank_descriptors_searched": int(bank_searched),
        "windowed_bank": bool(windowed_bank),
        "previous_node": None if previous_node is None else int(previous_node),
        "match_ratio_threshold": float(match_ratio),
        "accepted_pairs": int(len(points2d)),
        "accepted_pair_fraction": float(len(points2d) / max(query_count, 1)),
        "unique_matched_points": int(len(unique_owners)),
        "duplicate_pair_fraction": float(1.0 - len(unique_owners) / max(len(points2d), 1)),
        "best_distance": distribution_stats(best_distance[np.isfinite(best_distance)]),
        "second_distance": distribution_stats(second_distance[np.isfinite(second_distance)]),
        "accepted_best_distance": distribution_stats(best_distance[keep]),
        "accepted_second_distance": distribution_stats(second_distance[keep]),
        "accepted_ratio": distribution_stats(ratios[keep]),
        "pair_coverage": point_coverage(points2d, query["width"], query["height"]),
        "pair_points3d": points3d_geometry(points3d),
    }


def add_camera(diagnostics, image_size, map_image_size, source, camera_matrix, distortion):
    diagnostics["image_size"] = [int(value) for value in image_size]
    diagnostics["map_image_size"] = [int(value) for value in map_image_size]
    diagnostics["intrinsics_source"] = source
    diagnostics["K"] = camera_matrix.tolist()
    diagnostics["dist"] = np.asarray(distortion).reshape(-1).tolist()


def begin_pnp(diagnostics, max_error, confidence, iterations):
    diagnostics["pnp"] = {
        "reprojection_threshold_px": float(max_error),
        "confidence": float(confidence),
        "iterations": int(iterations),
        "method": "EPNP_RANSAC+RefineLM",
    }


def add_raw_pnp(diagnostics, ok, inliers, pair_count):
    count = 0 if inliers is None else int(len(inliers))
    diagnostics["pnp"]["raw_ok"] = bool(ok)
    diagnostics["pnp"]["raw_inliers"] = count
    diagnostics["pnp"]["raw_inlier_ratio"] = float(count / max(pair_count, 1))


def add_refined_pnp(diagnostics, points2d, points3d, owners, indices, rvec, tvec, camera_matrix, distortion, max_error):
    projected, _ = cv2.projectPoints(points3d[indices].astype(np.float64), rvec, tvec, camera_matrix, distortion)
    reprojection = np.linalg.norm(projected.reshape(-1, 2) - points2d[indices], axis=1)
    projected_all, _ = cv2.projectPoints(points3d.astype(np.float64), rvec, tvec, camera_matrix, distortion)
    reprojection_all = np.linalg.norm(projected_all.reshape(-1, 2) - points2d, axis=1)
    depths = (cv2.Rodrigues(rvec)[0] @ points3d[indices].T + tvec).T[:, 2]
    unique_owners = np.unique(owners[indices])
    diagnostics["pnp"].update(
        {
            "inliers": int(len(indices)),
            "inlier_ratio": float(len(indices) / max(len(points2d), 1)),
            "unique_inlier_points": int(len(unique_owners)),
            "duplicate_inlier_fraction": float(1.0 - len(unique_owners) / max(len(indices), 1)),
            "reprojection_error_px": distribution_stats(reprojection),
            "all_pair_reprojection_error_px": distribution_stats(reprojection_all),
            "refined_pairs_within_threshold": int(np.count_nonzero(reprojection_all <= max_error)),
            "refined_pair_inlier_ratio": float(np.mean(reprojection_all <= max_error)),
            "positive_depth_fraction": float(np.mean(depths > 0)),
            "inlier_coverage": point_coverage(
                points2d[indices], diagnostics["image_size"][0], diagnostics["image_size"][1]
            ),
            "inlier_points3d": points3d_geometry(points3d[indices]),
            "rvec": np.asarray(rvec).reshape(-1).tolist(),
            "tvec": np.asarray(tvec).reshape(-1).tolist(),
        }
    )


def add_pose(diagnostics, camera_center, forward, lead_center, command):
    diagnostics["pose"] = {
        "camera_center": camera_center.tolist(),
        "forward": forward.tolist(),
        "camera_center_lead": np.asarray(lead_center).tolist(),
        "node": int(command["node"]),
        "target_node": int(command.get("target_node", command["node"])),
        "bearing_deg": float(command["bearing_deg"]),
        "dist_to_route": float(command["dist_to_route"]),
        "offset": float(command["offset"]),
    }
