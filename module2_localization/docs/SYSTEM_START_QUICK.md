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

Дождаться сообщений о загрузке всех `7 + 3` шардов и двух полных recovery-карт. По умолчанию выбран
`Маршрут 1-2`, режим `dual`, ведущая камера — передняя.

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

В админке выбрать маршрут, включить или выключить светофор и нажать `СТАРТ`.

## Контроль

```bash
tegrastats
curl http://127.0.0.1:8080/api/status
```

Остановка ручного запуска: `Ctrl+C` в терминалах демона и админки.
