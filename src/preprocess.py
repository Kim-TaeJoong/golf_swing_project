import cv2
import os
import yt_dlp
from utils.utils import normalize_by_pelvis
import numpy as np  # 🔥 필터 적용을 위해 추가

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


# ---------------------------------------------------------
# 🌟 새롭게 추가한 화질 개선(Enhanced) 전처리 함수
# ---------------------------------------------------------
def process_swing_video_enhanced(input_path, output_path, start_frame, end_frame, crop_box=None, target_fps=30):
    """ 시간 자르기 + 크롭 + FPS 고정 + CLAHE(대비) + Sharpening(샤프닝)이 모두 적용된 함수 """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ 영상을 열 수 없습니다: {input_path}")
        return

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    if crop_box:
        x, y, w, h = crop_box
    else:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        x, y = 0, 0

    frame_step = max(1, int(source_fps / target_fps))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (w, h))

    print(f"🛠️ 고급 전처리 중: {start_frame}~{end_frame} 프레임 | {target_fps}FPS | 크롭 {w}x{h}")
    print("✨ 영상 화질 개선 필터(CLAHE & Sharpening) 적용 중...")

    # 필터 도구 준비
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    sharpen_kernel = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])

    current_idx = start_frame
    while current_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        if (current_idx - start_frame) % frame_step == 0:
            # 1. 크롭
            if crop_box:
                frame = frame[y:y+h, x:x+w]
            
            # 2. CLAHE (대비 최적화)
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

            # 3. Sharpening (샤프닝)
            frame = cv2.filter2D(frame, -1, sharpen_kernel)

            # 4. 저장
            out.write(frame)
            
        current_idx += 1

    cap.release()
    out.release()
    print(f"✅ 고급 전처리 완료: {output_path}")

# --- 사용 예시 ---
#if __name__ == "__main__":
    # 1. 다운로드 (필요시)
    # download_video("URL", "data/raw/tiger_raw.mp4")

    # 2. 통합 전처리 실행 (예: 타이거 우즈 0~25초, 정면 320x360 크롭, 30FPS 고정)
    # tiger_crop = (0, 0, 320, 360) # 640x360 영상의 왼쪽 절반
    # process_swing_video("data/raw/tiger_raw.mp4", "data/processed/tiger_final.mp4", 
    #                     start_frame=0, end_frame=750, crop_box=tiger_crop)

# 1. 타이거우즈 유튭 동영상 다운로드
    
'''if __name__ == "__main__":
    # 1. 다운로드할 유튜브 주소와 저장할 위치 설정
    video_url = "https://www.youtube.com/watch?v=Jlp8G9paliw"
    save_path = "data/raw/tiger_raw.mp4" 

    print(f"다운로드 시작: {video_url}")
    
    # 2. 다운로드 함수 실행!
    download_video(video_url, save_path)
    
    print(f"✅ 다운로드 완료! 파일 위치: {save_path}")
'''

# 2. tiger_raw process_swing_video 실행
'''
if __name__ == "__main__":
    # 1. 파일 경로를 다운로드 받은 이름(tiger_raw.mp4)으로 설정
    input_file = "data/raw/tiger_raw.mp4"
    output_file = "data/processed/tiger_final.mp4"

    # 2. 원본이 640x360이므로, 왼쪽 절반(정면 뷰)은 가로 320, 세로 360이 됩니다!
    tiger_crop = (0, 0, 320, 360)

    print("타이거 우즈 유튜브 영상 전처리 시작...")

    # 3. 통합 함수 실행 (0~25초 구간 자르기 + 정면 자르기 + 30FPS 고정 한 번에!)
    # 30FPS 기준 25초는 750프레임입니다.
    process_swing_video(
        input_path=input_file, 
        output_path=output_file, 
        start_frame=0, 
        end_frame=750, 
        crop_box=tiger_crop,
        target_fps=30 
    )
'''

# 화질 개선 함수 실행

if __name__ == "__main__":
    input_file = "data/raw/tiger_raw.mp4"
    output_file = "data/processed/tiger_final_enhanced.mp4"

    # 원본(640x360)의 왼쪽 절반 자르기
    tiger_crop = (0, 0, 320, 360) 

    print("타이거 우즈 영상 전처리 시작...")

    # 🔥 enhanced 함수를 호출!
    process_swing_video_enhanced(
        input_path=input_file, 
        output_path=output_file, 
        start_frame=0, 
        end_frame=750, 
        crop_box=tiger_crop,
        target_fps=30 
    )