# https://www.aipacommander.com/entry/2017/12/27/155711

#%%
import socket
import numpy as np
import cv2
import sys

#%%

if len(sys.argv) < 3:
    print("python main.py <cam ip> <viewer ip>")
    sys.exit()

ip_address = sys.argv[1]
cap = cv2.VideoCapture(f"http://{ip_address}:4747/video")
to_send_addr = (sys.argv[2], 9999)

#%%
size = 128
resize_size = 128
while True:
    _, frame = cap.read()
    ycenter = frame.shape[0] // 2
    xcenter = frame.shape[1] // 2
    frame = cv2.resize(frame[ycenter - size: ycenter + size, xcenter - size : xcenter + size], (resize_size, resize_size))

    jpg_str = cv2.imencode(".jpeg", frame)

    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for v in np.array_split(jpg_str[1], 10):
        udp.sendto(v.tostring(), to_send_addr)

    udp.sendto(b"__end__", to_send_addr)
    udp.close()
