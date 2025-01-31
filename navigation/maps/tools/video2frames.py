import cv2
import os
import argparse
from pathlib import Path


def get_frames(video_path, frame_interval):
    cap = cv2.VideoCapture(video_path)

    frames = []
    frame_cnt = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_cnt % frame_interval == 0:
            frames.append(frame)
        frame_cnt += 1
    
    cap.release()
    return frames


def main(args):
    map_id = args.map_id
    frame_interval = args.interval
    video_file = args.video_file
    start_id = args.start_id
    print(f"[*] map_id: {map_id}, frame_interval: {frame_interval}, video_file: {video_file}, start_id: {start_id}")

    maps_dir = Path(__file__).parent.parent
    map_dir = os.path.join(maps_dir, map_id)

    video_path = os.path.join(map_dir, "video", video_file)
    frames = get_frames(video_path=video_path, frame_interval=frame_interval)

    frames_dir = os.path.join(map_dir, "frames")
    os.makedirs(frames_dir, exist_ok=False)
    k = 0

    while (4 * k + 3) <= len(frames) - 1:
        path = os.path.join(frames_dir, f"{k+start_id}")
        os.makedirs(path, exist_ok=False)

        for i in range(4):
            cv2.imwrite(os.path.join(path, f"{i}.jpg"), frames[4*k+i])
        
        k += 1
    
    print("[*] Save completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--map-id", type=str, help="対象のマップID")
    parser.add_argument("-f", "--video-file", type=str, help="対象の動画のファイル名")
    parser.add_argument("-i", "--interval", type=int, help="抽出するフレームの間隔枚数(default: 1)", default=1)
    parser.add_argument("-s", "--start-id", type=int, help="最初に追加するノードのID(default: 0)", default=0)
    args = parser.parse_args()

    main(args)