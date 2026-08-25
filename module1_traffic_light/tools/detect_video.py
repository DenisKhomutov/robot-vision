import argparse
import sys
from pathlib import Path

import cv2
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from module1_traffic_light import init_models
from module1_traffic_light.core.classifier import get_classifier
from module1_traffic_light.core.detector import get_detector


def color_for(label):
    lo = label.lower()
    if "red" in lo or "крас" in lo:
        return (0, 0, 255)
    if "green" in lo or "зел" in lo:
        return (0, 200, 0)
    if "yellow" in lo or "жёл" in lo or "жел" in lo:
        return (0, 210, 255)
    return (0, 200, 255)


def main():
    ap = argparse.ArgumentParser(description="Прогон видео через пайплайн светофора: рамка + класс -> out")
    ap.add_argument("video")
    ap.add_argument("--out", default="module2_localization/out/traffic_preview")
    ap.add_argument("--step", type=int, default=5, help="обрабатывать каждый N-й кадр")
    ap.add_argument("--det-conf", type=float, default=None,
                    help="порог детекции светофора (ниже дефолтного 0.25 -> ловит больше/дальше)")
    ap.add_argument("--save-empty", action="store_true", help="сохранять и кадры без светофора")
    args = ap.parse_args()

    if args.det_conf is not None:
        from module1_traffic_light import config as tl_config
        tl_config.DET_CONF = args.det_conf
        print(f"[детекция] порог DET_CONF = {args.det_conf}", flush=True)

    init_models()
    det, cls = get_detector(), get_classifier()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"не открывается видео: {args.video}")
        return 1
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = saved = hits = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % args.step == 0:
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            boxes = det.detect(pil)
            for (x1, y1, x2, y2) in boxes:
                label, conf = cls.classify(pil.crop((x1, y1, x2, y2)))
                c = color_for(label)
                cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
                cv2.putText(frame, f"{label} {conf:.0%}", (x1, max(y1 - 8, 16)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2, cv2.LINE_AA)
            if boxes:
                hits += 1
            if boxes or args.save_empty:
                cv2.imwrite(str(out / f"{idx:06d}.jpg"), frame)
                saved += 1
            if idx % (args.step * 100) == 0:
                print(f"кадр {idx}/{total} | со светофором {hits} | сохранено {saved}", flush=True)
        idx += 1
    cap.release()
    print(f"\nготово: со светофором {hits}, сохранено {saved} кадров -> {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
