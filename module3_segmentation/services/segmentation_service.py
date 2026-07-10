from typing import Any

from PIL import Image

from ..core.segmentator import get_segmentator


def analyze(image: Image.Image) -> dict[str, Any]:
    segments = get_segmentator().segment(image)
    return {
        "segments": segments,
        "path_present": len(segments) > 0,
    }
