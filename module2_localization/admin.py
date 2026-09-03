"""Совместимый entrypoint веб-админки."""

from .runtime.admin_http import serve

__all__ = ["serve"]


if __name__ == "__main__":
    from .tools.admin_server import main

    raise SystemExit(main())
