import sys
import time
from pathlib import Path

import numpy as np
import torch

IMAGE0 = "experiment/images/current.jpg"
IMAGE1 = "experiment/images/reference.jpg"
RELOC3R_DIR = "experiment/reloc3r"
OUTDIR = "experiment/results_reloc3r"
MODEL_RES = "512"
FORCE_CPU = True
SHOW_3D = True

sys.path.insert(0, str(Path(RELOC3R_DIR).resolve()))


def _load_model_safe(res: int, device: torch.device, model_cls):
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    model = model_cls(img_size=res)
    path = hf_hub_download(f"siyan824/reloc3r-{res}", "model.safetensors")
    state_dict = {}
    with safe_open(path, framework="np") as f:
        for key in f.keys():
            state_dict[key] = torch.from_numpy(np.array(f.get_tensor(key)))
    model.load_state_dict(state_dict, strict=False)
    return model.to(device).eval()


def rotation_to_yaw_pitch_roll(rotation: np.ndarray) -> tuple[float, float, float]:
    yaw = float(np.degrees(np.arctan2(rotation[0, 2], rotation[2, 2])))
    pitch = float(np.degrees(np.arcsin(-rotation[1, 2])))
    roll = float(np.degrees(np.arctan2(rotation[1, 0], rotation[1, 1])))
    return yaw, pitch, roll


def _camera_frustum_world(cam_to_world: np.ndarray, w: int, h: int, focal: float, scale: float) -> np.ndarray:
    k = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]], dtype=float)
    corners_px = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=float)
    rays = (np.linalg.inv(k) @ corners_px.T).T
    rays = rays / np.linalg.norm(rays, axis=1, keepdims=True) * scale
    pts_cam = np.vstack([np.zeros(3), rays])
    return (cam_to_world[:3, :3] @ pts_cam.T).T + cam_to_world[:3, 3]


def _draw_camera(ax, cam_to_world: np.ndarray, color: str, label: str, scale: float = 0.3) -> None:
    p = _camera_frustum_world(cam_to_world, 640, 480, 525.0, scale)
    center, corners = p[0], p[1:]
    for c in corners:
        ax.plot(*zip(center, c), color=color, linewidth=1.2)
    loop = np.vstack([corners, corners[0]])
    ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], color=color, linewidth=1.2)
    ax.scatter(*center, color=color, s=40)
    ax.text(*center, f"  {label}", color=color, fontsize=10, weight="bold")


def _set_equal_aspect(ax, pts: np.ndarray) -> None:
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    center = (lo + hi) / 2
    r = (hi - lo).max() / 2 + 1e-6
    ax.set_xlim(center[0] - r, center[0] + r)
    ax.set_ylim(center[1] - r, center[1] + r)
    ax.set_zlim(center[2] - r, center[2] + r)


def visualize_pose(pose2to1: np.ndarray, yaw: float, pitch: float, roll: float, outdir: Path, show: bool) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    pose_current = np.eye(4)  # view1 = current в начале координат
    pose_reference = pose2to1  # view2 = reference относительно current

    _draw_camera(ax, pose_current, "green", "current (v1)")
    _draw_camera(ax, pose_reference, "orange", "reference (v2)")

    for vec, col in [((1, 0, 0), "red"), ((0, 1, 0), "limegreen"), ((0, 0, 1), "blue")]:
        ax.plot(*zip((0, 0, 0), np.array(vec) * 0.4), color=col, linewidth=2)

    all_pts = np.vstack([
        _camera_frustum_world(pose_current, 640, 480, 525.0, 0.3),
        _camera_frustum_world(pose_reference, 640, 480, 525.0, 0.3),
    ])
    _set_equal_aspect(ax, all_pts)
    ax.set_xlabel("X (право)")
    ax.set_ylabel("Y (низ)")
    ax.set_zlabel("Z (вперёд)")
    ax.set_title(f"reloc3r: yaw={yaw:+.1f}°  pitch={pitch:+.1f}°  roll={roll:+.1f}°")
    ax.view_init(elev=-70, azim=-90)

    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "reloc3r_pose3d.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"[INFO] 3D-визуализация сохранена: {out}")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    device = torch.device("cpu" if FORCE_CPU or not torch.cuda.is_available() else "cuda")
    print(f"[INFO] device = {device}")

    try:
        from reloc3r.reloc3r_relpose import Reloc3rRelpose, inference_relpose
        from reloc3r.utils.device import to_numpy
        from reloc3r.utils.image import check_images_shape_format, load_images
    except ImportError as e:
        print(f"[ERROR] Не найден пакет reloc3r: {e}")
        print("Проверь RELOC3R_DIR и что подмодуль croco инициализирован:")
        print("  cd experiment/reloc3r && git submodule update --init --recursive")
        sys.exit(1)

    t0 = time.perf_counter()
    model = _load_model_safe(int(MODEL_RES), device, Reloc3rRelpose)
    print(f"[INFO] Reloc3r загружен за {time.perf_counter() - t0:.2f}с")

    images = load_images([IMAGE0, IMAGE1], size=int(MODEL_RES))
    images = check_images_shape_format(images, device)
    batch = [images[0], images[1]]

    t0 = time.perf_counter()
    pose2to1 = to_numpy(inference_relpose(batch, model, device)[0])
    t_infer = time.perf_counter() - t0

    rotation = pose2to1[:3, :3]
    translation = pose2to1[:3, 3]
    t_norm = translation / (np.linalg.norm(translation) + 1e-9)

    yaw, pitch, roll = rotation_to_yaw_pitch_roll(rotation)

    print(f"[TIMING] inference: {t_infer * 1000:.0f}мс")
    print(f"[POSE] yaw={yaw:+.1f}° pitch={pitch:+.1f}° roll={roll:+.1f}°")
    print(f"[POSE] направление сдвига (ед. вектор): x={t_norm[0]:+.2f} y={t_norm[1]:+.2f} z={t_norm[2]:+.2f}")

    dead_zone = 3.0
    if yaw > dead_zone:
        hint = "чтобы попасть в reference, руль ВПРАВО"
    elif yaw < -dead_zone:
        hint = "чтобы попасть в reference, руль ВЛЕВО"
    else:
        hint = "ориентация совпадает с reference, прямо"
    print(f"[RESULT] {hint}")

    visualize_pose(pose2to1, yaw, pitch, roll, Path(OUTDIR), SHOW_3D)


if __name__ == "__main__":
    main()
