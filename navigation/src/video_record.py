import cv2

url = "http://192.168.11.12:4747/video"
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("[Error] Can't open the camera.")
    exit()

fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
print(fourcc)

fps = int(cap.get(cv2.CAP_PROP_FPS))
print(fps)

frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print(frame_size)

#out = cv2.VideoWriter('output.avi', fourcc, fps, frame_size)