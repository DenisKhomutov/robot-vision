import socket
import struct
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

TOPIC = "/unilidar/cloud"
PORT = 9099


class Srv(Node):
    def __init__(self):
        super().__init__("lidar_srv")
        self.create_subscription(PointCloud2, TOPIC, self.cb, 10)
        self.lock = threading.Lock()
        self.buf = None

    def cb(self, m):
        pts = point_cloud2.read_points(m, field_names=("x", "y", "z"), skip_nans=True)
        xyz = np.array([[p[0], p[1], p[2]] for p in pts], dtype=np.float32)
        with self.lock:
            self.buf = xyz.tobytes()

    def latest(self):
        with self.lock:
            return self.buf


def serve(node):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", PORT))
    s.listen(1)
    print(f"жду клиента на :{PORT} ...", flush=True)
    while True:
        conn, addr = s.accept()
        print("клиент подключился", addr, flush=True)
        try:
            while True:
                b = node.latest()
                if b:
                    conn.sendall(struct.pack("<I", len(b)) + b)
                threading.Event().wait(0.08)
        except (BrokenPipeError, ConnectionResetError):
            print("клиент отключился", flush=True)
            conn.close()


def main():
    rclpy.init()
    node = Srv()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    serve(node)


if __name__ == "__main__":
    main()
