"""Anchor a front-camera COLMAP model to a synchronized rear-camera route.

The front and rear videos must use original video-frame numbers in image names.
No feature correspondence between the cameras is required: rear poses are
interpolated in time and the constant rig rotation is estimated from the two
independently reconstructed orientation sequences.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pycolmap
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp


def _stamp(name: str) -> tuple[int, int]:
    stem = Path(name).stem
    if stem.startswith("p2_"):
        return 2, int(stem.split("_", 1)[1])
    if stem.startswith(("p1_", "p1b_")):
        return 1, int(stem.split("_", 1)[1])
    return 1, int(stem)


def _camera_to_world(image: pycolmap.Image) -> tuple[np.ndarray, np.ndarray]:
    pose = image.cam_from_world().inverse()
    return np.asarray(pose.translation), np.asarray(pose.rotation.matrix())


def _rear_samples(reconstruction: pycolmap.Reconstruction, part: int):
    samples = []
    for image in reconstruction.images.values():
        image_part, frame = _stamp(image.name)
        if image_part == part:
            center, rotation = _camera_to_world(image)
            samples.append((frame, center, rotation))
    samples.sort(key=lambda item: item[0])

    unique = {}
    for frame, center, rotation in samples:
        unique[frame] = (center, rotation)
    return [(frame, *unique[frame]) for frame in sorted(unique)]


def _interpolate(samples, frames: np.ndarray):
    times = np.asarray([item[0] for item in samples], dtype=float)
    centers = np.asarray([item[1] for item in samples])
    rotations = Rotation.from_matrix(np.asarray([item[2] for item in samples]))
    interp_centers = np.column_stack(
        [np.interp(frames, times, centers[:, axis]) for axis in range(3)]
    )
    interp_rotations = Slerp(times, rotations)(frames).as_matrix()
    return interp_centers, interp_rotations


def _fit_world_rotation(front_centers: np.ndarray, rear_centers: np.ndarray) -> np.ndarray:
    """Robustly fit Q such that rear direction ~= Q @ front direction."""
    stride = min(10, max(1, len(front_centers) // 20))
    front_delta = front_centers[stride:] - front_centers[:-stride]
    rear_delta = rear_centers[stride:] - rear_centers[:-stride]
    front_norm = np.linalg.norm(front_delta, axis=1)
    rear_norm = np.linalg.norm(rear_delta, axis=1)
    keep = (front_norm > np.quantile(front_norm, 0.3)) & (
        rear_norm > np.quantile(rear_norm, 0.3)
    )
    front_dirs = front_delta[keep] / front_norm[keep, None]
    rear_dirs = rear_delta[keep] / rear_norm[keep, None]

    inliers = np.ones(len(front_dirs), dtype=bool)
    world_rotation = np.eye(3)
    for _ in range(6):
        covariance = rear_dirs[inliers].T @ front_dirs[inliers]
        u, _, vt = np.linalg.svd(covariance)
        world_rotation = u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt
        cosines = np.sum((front_dirs @ world_rotation.T) * rear_dirs, axis=1)
        errors = np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))
        threshold = max(8.0, float(np.quantile(errors, 0.65)))
        inliers = errors <= threshold
    return world_rotation


def _fit_rig_rotations(
    world_rotation: np.ndarray,
    front_rotations: np.ndarray,
    rear_rotations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    relative = rear_rotations.transpose(0, 2, 1) @ world_rotation @ front_rotations
    rig_rotation = Rotation.from_matrix(relative).mean().as_matrix()

    initial = np.r_[
        Rotation.from_matrix(world_rotation).as_rotvec(),
        Rotation.from_matrix(rig_rotation).as_rotvec(),
    ]

    def residual(parameters):
        q = Rotation.from_rotvec(parameters[:3]).as_matrix()
        rig = Rotation.from_rotvec(parameters[3:]).as_matrix()
        predicted = q @ front_rotations
        target = rear_rotations @ rig
        delta = target.transpose(0, 2, 1) @ predicted
        return Rotation.from_matrix(delta).as_rotvec().ravel()

    result = least_squares(residual, initial, loss="huber", f_scale=np.deg2rad(1.0))
    q = Rotation.from_rotvec(result.x[:3]).as_matrix()
    rig = Rotation.from_rotvec(result.x[3:]).as_matrix()
    errors = np.linalg.norm(residual(result.x).reshape(-1, 3), axis=1)
    return q, rig, np.degrees(errors)


def anchor(
    front_model: Path,
    rear_model: Path,
    output: Path,
    offsets: dict[int, int],
) -> None:
    front = pycolmap.Reconstruction(front_model)
    rear = pycolmap.Reconstruction(rear_model)

    records = []
    for part in (1, 2):
        rear_part = _rear_samples(rear, part)
        first_frame, last_frame = rear_part[0][0], rear_part[-1][0]
        front_part = []
        for image in front.images.values():
            image_part, frame = _stamp(image.name)
            rear_frame = frame + offsets[part]
            if image_part == part and first_frame <= rear_frame <= last_frame:
                center, rotation = _camera_to_world(image)
                front_part.append((frame, rear_frame, image.image_id, center, rotation))
        front_part.sort(key=lambda item: item[0])
        frames = np.asarray([item[1] for item in front_part], dtype=float)
        rear_centers, rear_rotations = _interpolate(rear_part, frames)
        for item, rear_center, rear_rotation in zip(
            front_part, rear_centers, rear_rotations, strict=True
        ):
            frame, _, image_id, front_center, front_rotation = item
            records.append(
                (part, frame, image_id, front_center, front_rotation, rear_center, rear_rotation)
            )

    records.sort(key=lambda item: (item[0], item[1]))
    front_centers = np.asarray([item[3] for item in records])
    front_rotations = np.asarray([item[4] for item in records])
    rear_centers = np.asarray([item[5] for item in records])
    rear_rotations = np.asarray([item[6] for item in records])

    q0 = _fit_world_rotation(front_centers, rear_centers)
    _, rig_rotation, rotation_errors = _fit_rig_rotations(
        q0, front_rotations, rear_rotations
    )
    anchored_ids = {item[2] for item in records}


    for point3d_id in list(front.point3D_ids()):
        front.delete_point3D(point3d_id)

    for image in list(front.images.values()):
        if image.image_id not in anchored_ids and image.has_pose:
            front.deregister_frame(image.frame_id)

    for item in records:
        _, _, image_id, _, _, center, rear_rotation = item
        camera_to_world_rotation = rear_rotation @ rig_rotation
        world_to_camera_rotation = camera_to_world_rotation.T
        translation = -world_to_camera_rotation @ center
        pose = pycolmap.Rigid3d(
            pycolmap.Rotation3d(world_to_camera_rotation), translation
        )
        image = front.images[image_id]
        front.frames[image.frame_id].rig_from_world = pose

    output.mkdir(parents=True, exist_ok=True)
    front.write(output)
    quantiles = np.quantile(rotation_errors, [0.5, 0.9, 1.0])
    print(f"Anchored images: {len(anchored_ids)} / {front.num_images()}")
    print(
        "Rig rotation residual (median/p90/max): "
        f"{quantiles[0]:.3f} / {quantiles[1]:.3f} / {quantiles[2]:.3f} deg"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--front-model", type=Path, required=True)
    parser.add_argument("--rear-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offset-p1", type=int, default=0)
    parser.add_argument("--offset-p2", type=int, default=0)
    args = parser.parse_args()
    anchor(
        args.front_model,
        args.rear_model,
        args.output,
        {1: args.offset_p1, 2: args.offset_p2},
    )


if __name__ == "__main__":
    main()
