"""Совместимый entrypoint сервиса локализации."""

from .runtime.application import run

__all__ = ["run"]


if __name__ == "__main__":
    from .tools.run_localization import main

    raise SystemExit(main())
