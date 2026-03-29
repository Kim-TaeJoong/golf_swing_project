import cv2
import os
import yt_dlp
from utils.utils import normalize_by_pelvis

def download_video(url, output_path):
    ydl_opts = {
        'outtmpl' : output_path,
        'format': 'best[height<=1080]'
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

def trim_video(input_path, output_path, start_frame, end_frame):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("영상을 열 수 없습니다.")
        return

    source_fps = cap.get(cv2.CAP_PROP_FPS)
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

import cv2

def crop_video_by_area(input_path, output_path, x, y, w, h):
    """
    원본 영상(640x360 등)에서 특정 좌표(x, y)를 기준으로 
    가로 w, 세로 h 만큼의 영역만 잘라내어 저장합니다.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ 영상을 열 수 없습니다: {input_path}")
        return

    # 1. 원본 정보 획득 (FPS 등)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 2. VideoWriter 설정
    # 핵심: 출력 해상도를 반드시 '자른 후의 크기(w, h)'로 설정해야 오류가 없습니다.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    print(f"📐 크롭 시작: 원본(640x360) -> 대상({w}x{h})")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 3. 이미지 슬라이싱 (Numpy Slicing)
            # frame[세로범위, 가로범위] 순서입니다.
            cropped_frame = frame[y:y+h, x:x+w]
            
            # 4. 프레임 기록
            out.write(cropped_frame)
            
        print(f"✅ 크롭 완료: {output_path}")
    
    finally:
        # 자원 해제 (파일 손상 방지)
        cap.release()
        out.release()

def process_swing_video(input_path, output_path, start_frame, end_frame, crop_box=None, target_fps=30):
    """
    1. 시간 자르기 (Trim)
    2. 속도 표준화 (30FPS 고정)
    3. 공간 자르기 (Crop)
    를 한 번에 수행합니다.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ 영상을 열 수 없습니다: {input_path}")
        return

    # 1. 원본 정보 획득
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # 2. 크롭 영역 설정에 따른 해상도 결정
    if crop_box:
        x, y, w, h = crop_box
    else:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        x, y = 0, 0

    # 3. 30FPS 변환을 위한 프레임 스킵 간격 계산
    # 예: 240fps 영상이라면 8프레임마다 1장씩 선택 (240/30 = 8)
    frame_step = max(1, int(source_fps / target_fps))
    
    # 4. VideoWriter 설정 (target_fps와 크롭된 w, h 사용)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (w, h))

    print(f"전처리 중: {start_frame} ~ {end_frame} 프레임 | {target_fps}FPS | 크롭 {w}x{h}")

    current_idx = start_frame
    while current_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 30FPS를 맞추기 위해 지정된 스텝만큼만 저장
        if (current_idx - start_frame) % frame_step == 0:
            if crop_box:
                frame = frame[y:y+h, x:x+w]
            out.write(frame)
            
        current_idx += 1

    cap.release()
    out.release()
    print(f"✅ 전처리 완료: {output_path}")

# --- 사용 예시 ---
#if __name__ == "__main__":
    # 1. 다운로드 (필요시)
    # download_video("URL", "data/raw/tiger_raw.mp4")

    # 2. 통합 전처리 실행 (예: 타이거 우즈 0~25초, 정면 320x360 크롭, 30FPS 고정)
    # tiger_crop = (0, 0, 320, 360) # 640x360 영상의 왼쪽 절반
    # process_swing_video("data/raw/tiger_raw.mp4", "data/processed/tiger_final.mp4", 
    #                     start_frame=0, end_frame=750, crop_box=tiger_crop)