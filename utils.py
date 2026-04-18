import socket
import threading
import cv2
import os
import time
from concurrent.futures import ThreadPoolExecutor

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def scan_droidcam(callback):
    """Scans the local subnet for port 4747 (DroidCam) and 8080 (IP Cam)"""
    local_ip = get_local_ip()
    if local_ip == "127.0.0.1":
        return

    base_ip = ".".join(local_ip.split(".")[:-1]) + "."

    def check_ip(ip, port):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                if s.connect_ex((ip, port)) == 0:
                    if port == 4747:
                        url = f"http://{ip}:4747/video"
                    else:
                        url = f"http://{ip}:{port}/video"
                    callback(url)
        except Exception:
            pass

    tasks = [(base_ip + str(i), port) for port in [4747, 8080] for i in range(1, 255)]
    with ThreadPoolExecutor(max_workers=64) as pool:
        futures = [pool.submit(check_ip, ip, port) for ip, port in tasks]
        for f in futures:
            try:
                f.result(timeout=3.0)
            except Exception:
                pass

def list_local_cameras():
    """Returns a list of actually available camera indices"""
    cams = []
    for i in range(10):
        if os.path.exists(f"/dev/video{i}"):
            cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
            if cap.isOpened():
                cams.append(str(i))
                cap.release()
    return cams
