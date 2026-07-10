def init_models() -> None:
    from .core.segmentator import init_segmentator

    init_segmentator()
