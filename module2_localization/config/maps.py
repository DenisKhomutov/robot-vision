"""Реестр карт и параметры, привязанные к конкретным маршрутам."""

# Карта по умолчанию намеренно отсутствует. Runtime-карты являются артефактами
# деплоя и регистрируются здесь только после полевой проверки.
# Schema: {"route-id": {"label": "Display name", "map": "relative/map/path"}}
ROUTES = {}
DEFAULT_ROUTE = None
DEFAULT_MAP = None

# Параметры зон индексируются именем карты/маршрута.
TRAFFIC_ZONES = {}
BACKWARD_ZONES = {}
BACKWARD_MAPS = set()
BACKWARD_RIGHT_ONLY_MAPS = set()
STEERING_OUTLIER_GUARDS = {}


__all__ = [name for name in globals() if name.isupper()]

