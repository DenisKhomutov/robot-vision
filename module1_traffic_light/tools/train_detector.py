"""Дообучение детектора рамки светофора (best_det.pt) на новом датасете.

Датасет — стандартный YOLO detect формат, готовится и экспортируется отдельно
(Roboflow/CVAT/LabelImg и т.п.), сюда передаётся только путь к data.yaml:

    dataset_root/
      images/train/*.jpg
      images/val/*.jpg
      labels/train/*.txt   # class x_center y_center width height, нормировано 0..1
      labels/val/*.txt
      data.yaml

data.yaml:
    path: /abs/path/to/dataset_root
    train: images/train
    val: images/val
    names:
      0: traffic_light

Запуск:
    uv run --no-sync python -m module1_traffic_light.tools.train_detector \\
        --data /path/to/dataset_root/data.yaml
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from module1_traffic_light import config


def main():
    ap = argparse.ArgumentParser(description="Дообучение детектора светофора")
    ap.add_argument("--data", required=True, help="путь к data.yaml датасета")
    ap.add_argument("--weights", default=config.DETECTOR_WEIGHTS,
                    help="стартовые веса (по умолчанию текущий best_det.pt -> дообучение)")
    ap.add_argument("--epochs", type=int, default=60, help="потолок; реально остановит --patience")
    ap.add_argument("--imgsz", type=int, default=config.DET_IMGSZ)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=10,
                    help="ранняя остановка, эпох без улучшения val (меньше -> для узкого датасета)")
    ap.add_argument("--freeze", type=int, default=10,
                    help="заморозить первые N слоёв backbone (общие признаки не трогаем, "
                         "дообучаем только специфичные под новый маршрут слои); 0 = не морозить")
    ap.add_argument("--lr0", type=float, default=0.001,
                    help="стартовый learning rate (ниже, чем при обучении с нуля ~0.01, "
                         "чтобы не увести веса далеко от исходной точки)")
    ap.add_argument("--device", default=None, help="cuda/cpu/0,1,...; по умолчанию auto")
    ap.add_argument("--project", default=str(Path(config.DETECTOR_WEIGHTS).resolve().parents[1] / "runs" / "detect"))
    ap.add_argument("--name", default="finetune")
    ap.add_argument("--promote", action="store_true",
                    help="скопировать лучший результат в weights/best_det.pt после обучения")
    args = ap.parse_args()

    from ultralytics import YOLO
    import torch

    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    print(f"веса: {args.weights}")
    print(f"датасет: {args.data}")
    print(f"устройство: {device}, epochs={args.epochs} (потолок), patience={args.patience}, "
          f"imgsz={args.imgsz}, batch={args.batch}, freeze={args.freeze}, lr0={args.lr0}")

    model = YOLO(args.weights)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        freeze=args.freeze if args.freeze > 0 else None,
        lr0=args.lr0,
        device=device,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nобучение завершено -> {best}")
    if not best.exists():
        print("ОШИБКА: best.pt не найден")
        return 1

    if args.promote:
        dst = Path(config.DETECTOR_WEIGHTS)
        backup = dst.with_suffix(".pt.bak")
        if dst.exists():
            shutil.copy2(dst, backup)
            print(f"старые веса сохранены в {backup}")
        shutil.copy2(best, dst)
        print(f"новые веса установлены в {dst}")
    else:
        print(f"чтобы применить: cp {best} {config.DETECTOR_WEIGHTS}")
        print("(или перезапусти с --promote)")
    return 0


if __name__ == "__main__":
    sys.exit(main())