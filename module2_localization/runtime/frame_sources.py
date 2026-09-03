import asyncio
import os
import shutil
import subprocess
import threading

import cv2
import numpy as np


class ShmSource:
    def __init__(self, socket_path, width, height, fps):
        gst = shutil.which("gst-launch-1.0")
        if gst is None:
            raise RuntimeError("gst-launch-1.0 не найден")
        if not os.path.exists(socket_path):
            raise RuntimeError(f"нет сокета {socket_path}: fan-out")
        self.w, self.h = width, height
        self.frame_size = width * height * 3
        caps = f"video/x-raw,format=I420,width={width},height={height},framerate={fps}/1"
        read_fd, write_fd = os.pipe()
        os.set_inheritable(write_fd, True)
        cmd = [
            gst,
            "-q",
            "shmsrc",
            f"socket-path={socket_path}",
            "is-live=true",
            "do-timestamp=true",
            "!",
            caps,
            "!",
            "videoconvert",
            "!",
            "video/x-raw,format=BGR",
            "!",
            "fdsink",
            f"fd={write_fd}",
        ]
        self.proc = subprocess.Popen(cmd, pass_fds=(write_fd,), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.close(write_fd)
        self.reader = os.fdopen(read_fd, "rb", buffering=0)
        self._latest = None
        self._stamp = 0
        self.done = False
        self._stop = False
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()

    def _read_exactly(self, size):
        buf = bytearray()
        while len(buf) < size and not self._stop:
            chunk = self.reader.read(size - len(buf))
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf) if len(buf) == size else None

    def _loop(self):
        while not self._stop:
            raw = self._read_exactly(self.frame_size)
            if raw is None:
                self.done = True
                break
            frame = np.frombuffer(raw, np.uint8).reshape(self.h, self.w, 3)
            with self._lock:
                self._latest = frame
                self._stamp += 1

    def latest(self):
        with self._lock:
            return self._stamp, self._latest

    def stop(self):
        self._stop = True
        self.proc.terminate()
        try:
            self.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.reader.close()
        if self._thread.is_alive():
            self._thread.join(timeout=3.0)


class CameraSource:
    def __init__(self, spec):
        if isinstance(spec, str) and not spec.isdigit():
            self.cap = cv2.VideoCapture(spec, cv2.CAP_GSTREAMER)
        else:
            self.cap = cv2.VideoCapture(int(spec))
        if not self.cap.isOpened():
            raise RuntimeError(f"не открывается камера: {spec}")
        self._latest = None
        self._stamp = 0
        self.done = False
        self._stop = False
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()

    def _loop(self):
        while not self._stop:
            ok, frame = self.cap.read()
            if not ok:
                self.done = True
                break
            with self._lock:
                self._latest = frame
                self._stamp += 1
        self.cap.release()

    def latest(self):
        with self._lock:
            return self._stamp, self._latest

    def stop(self):
        self._stop = True
        if self._thread.is_alive():
            self._thread.join(timeout=3.0)


class VideoFileSource:
    def __init__(self, path: str, step: int = 2, fps: float = 30.0, loop: bool = True):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"не открывается видео: {path}")
        self.step = step
        self.dt = 1.0 / fps
        self.loop = bool(loop)
        self._latest = None
        self._stamp = 0
        self._stop = False
        self.cycle = 0
        self.done = False

    def start(self):
        asyncio.ensure_future(self._run())

    async def _run(self):
        idx = 0
        while not self._stop:
            ok, frame = self.cap.read()
            if not ok:
                if not self.loop:
                    self.done = True
                    self.cap.release()
                    return
                if not self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.path)
                    if not self.cap.isOpened():
                        self.done = True
                        return
                self.cycle += 1
                idx = 0
                await asyncio.sleep(0)
                continue
            if idx % self.step == 0:
                self._latest = frame
                self._stamp += 1
            idx += 1
            await asyncio.sleep(self.dt)

    def latest(self):
        return self._stamp, self._latest

    def stop(self):
        self._stop = True
        self.cap.release()
