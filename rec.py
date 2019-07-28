# https://www.aipacommander.com/entry/2017/12/27/155711

import socket
import numpy as np
import cv2


#%%
def recive(udp):
    buff = 1024 * 64
    while True:
        recive_data = bytes()
        while True:
            jpg_str, addr = udp.recvfrom(buff)
            is_len = len(jpg_str) == 7
            is_end = jpg_str == b"__end__"
            if is_len and is_end: break
            recive_data += jpg_str

        if len(recive_data) == 0: continue

        narray = np.fromstring(recive_data, dtype="uint8")

        img = cv2.imdecode(narray, 1)
        yield img


udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp.bind(("0.0.0.0", 9999))

for img in recive(udp):
    cv2.imshow("img", cv2.resize(img, (512, 512)))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
