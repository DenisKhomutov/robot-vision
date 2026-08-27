# Быстрый запуск всей системы на Jetson

```bash
cd ~/projects/robot-vision
```

## 1. Камеры и NATS

```bash
sudo systemctl start camera-fanout.service
docker start nats 2>/dev/null || docker run -d --name nats -p 4222:4222 --restart unless-stopped nats:latest
ls -l /tmp/cam_front_raw /tmp/cam_raw
```

## 2. Демон навигации

В отдельном терминале:

```bash
cd ~/projects/robot-vision
python -m module2_localization.service --shm
```

Демон запускается без выбранного маршрута (`DEFAULT_ROUTE=None`) и публикует безопасный
`stop` с `reason=route_not_selected`. Карты в этот момент память не занимают. Режим камер
по умолчанию — `front`; передняя камера также используется детектором светофора.

## 3. Веб-админка

В отдельном терминале:

```bash
cd ~/projects/robot-vision
python -m module2_localization.admin \
  --bind 0.0.0.0 --port 8080 \
  --nats-url nats://127.0.0.1:4222
```

Открыть с ноутбука или телефона:

```text
http://192.168.40.48:8080/
```

В админке выбрать маршрут и дождаться `route_loaded`, при необходимости включить или
выключить светофор и конечные манёвры 2-1, затем отдельно нажать `СТАРТ`. При выборе
маршрута загружаются его стартовый шард и полная recovery-карта; соседний шард
предзагружается во время движения.

## Контроль

```bash
tegrastats
curl http://127.0.0.1:8080/api/status
```

Остановка ручного запуска: `Ctrl+C` в терминалах демона и админки.
