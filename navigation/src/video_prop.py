import cv2

file_path = "../maps/0/video/0.mov"
cap = cv2.VideoCapture(file_path)

fps = cap.get(cv2.CAP_PROP_FPS)
print(f"[*]FPS: {fps} fps")

frame_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print(f"[*]Frame Num: {frame_num}")

video_time = frame_num / fps
print(f"[*]Video Time: {video_time} sec")