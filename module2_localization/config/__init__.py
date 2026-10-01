"""Единый совместимый интерфейс конфигурации runtime.

Существующий код продолжает обращаться к ``module2_localization.config``;
настройки физически разделены по назначению.
"""

from .deployment import *
from .maps import *
from .parameters import *

