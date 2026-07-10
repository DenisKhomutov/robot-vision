def init_models() -> None:
    from .core.classifier import init_classifier
    from .core.detector import init_detector

    init_detector()
    init_classifier()
