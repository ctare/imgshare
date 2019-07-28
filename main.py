#%%
import cv2

#%%
cap = cv2.VideoCapture("http://10.202.99.50:4747/video")

#%%
# 480 640
while True:
    _, frame = cap.read()
    ycenter = frame.shape[0] // 2
    xcenter = frame.shape[1] // 2
    size = 128
    frame = cv2.resize(frame[ycenter - size: ycenter + size, xcenter - size : xcenter + size], (128, 128))
    cv2.imshow("img", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
