import socket
import threading
import cv2
import os

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

def scan_droidcam(callback):
    """Scans the local subnet for port 4747 (DroidCam)"""
    local_ip = get_local_ip()
    if local_ip == "127.0.0.1":
        return
        
    base_ip = ".".join(local_ip.split(".")[:-1]) + "."
    
    def check_ip(ip):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.1)
            if s.connect_ex((ip, 4747)) == 0:
                url = f"http://{ip}:4747/video"
                callback(url)

    threads = []
    for i in range(1, 255):
        t = threading.Thread(target=check_ip, args=(base_ip + str(i),))
        t.start()
        threads.append(t)
    
    for t in threads:
        t.join(timeout=0.01) # Rapid fire join

def list_local_cameras():
    """Returns a list of available /dev/videoX indices"""
    cams = []
    for i in range(10):
        if os.path.exists(f"/dev/video{i}"):
            cams.append(str(i))
    return cams
