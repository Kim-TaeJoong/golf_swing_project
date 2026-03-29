import cv2
import os
import yt_dlp
from utils import normalize_by_pelvis

def download_video(url, output_path):
    ydl_opts = {
        'outtmpl' : output_path,
        'format' : 'bestvideo[height<=1080]+bestaudio/best'
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

def trim_video(input_path, output_path, start_frame, end_frame):
    cap = cv2.VideoCapture(input_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # 시작 프레임으로 이동
    
    # VideoWriter 로 잘라낸 구간만 새 파일로 저장
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for frame_num in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()