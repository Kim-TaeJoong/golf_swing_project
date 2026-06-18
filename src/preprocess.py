"""
preprocess.py — 파이프라인 1단계: 영상 전처리

역할:
  유튜브에서 골프 스윙 영상을 다운로드하고, 분석에 필요한 구간만 잘라내어
  MediaPipe가 정확하게 뼈대를 추출할 수 있도록 화질을 개선한 뒤 저장한다.

처리 순서:
  1. download_video        : yt-dlp로 유튜브 영상 다운로드
  2. trim_video            : 필요한 프레임 구간만 잘라내기
  3. crop_video_by_area    : 화면에서 선수만 나오는 영역으로 크롭
  4. process_swing_video_enhanced : 위 과정 + FPS 고정 + CLAHE + Sharpening 한 번에 처리
"""

import cv2
import os
import yt_dlp
from utils.utils import normalize_by_pelvis
import numpy as np


# ── 1. 다운로드 ────────────────────────────────────────────────────────────────
def download_video(url, output_path):
    """
    유튜브 URL에서 영상을 다운로드하여 output_path에 저장한다.

    Args:
        url         : 다운로드할 유튜브 영상 URL
        output_path : 저장할 파일 경로 (예: 'data/raw/tiger_raw.mp4')
    """
    ydl_opts = {
        'outtmpl' : output_path,
        'format': 'best[height<=1080]'  # 1080p 이하 최고 화질 선택
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


# ── 2. 시간 구간 자르기 ────────────────────────────────────────────────────────
def trim_video(input_path, output_path, start_frame, end_frame):
    """
    원본 영상에서 start_frame ~ end_frame 구간만 잘라내어 새 파일로 저장한다.

    Args:
        input_path  : 원본 영상 경로
        output_path : 잘라낸 영상을 저장할 경로
        start_frame : 시작 프레임 번호 (0-based)
        end_frame   : 종료 프레임 번호 (exclusive)
    """
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("영상을 열 수 없습니다.")
        return

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # 시작 프레임으로 이동

    fps    = cap.get(cv2.CAP_PROP_FPS)
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


# ── 3. 영역 크롭 ───────────────────────────────────────────────────────────────
import cv2

def crop_video_by_area(input_path, output_path, x, y, w, h):
    """
    원본 영상에서 (x, y)를 좌상단 기준으로 가로 w × 세로 h 영역만 잘라내어 저장한다.

    카메라 프레임에 두 선수가 모두 담겨 있을 때, 정면 뷰만 추출하거나
    배경을 제거하여 MediaPipe 인식 정확도를 높이는 데 사용한다.

    Args:
        input_path  : 원본 영상 경로
        output_path : 크롭된 영상 저장 경로
        x, y        : 크롭 영역의 좌상단 좌표 (픽셀)
        w, h        : 크롭 영역의 가로·세로 크기 (픽셀)
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"영상을 열 수 없습니다: {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    # 출력 해상도를 반드시 '자른 후의 크기(w, h)'로 설정해야 오류가 없다.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    print(f"크롭 시작: 원본(640x360) -> 대상({w}x{h})")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # NumPy 슬라이싱: frame[세로범위, 가로범위] 순서
            cropped_frame = frame[y:y+h, x:x+w]
            out.write(cropped_frame)

        print(f"크롭 완료: {output_path}")

    finally:
        # 자원 해제 — 이 블록이 없으면 저장된 영상이 손상될 수 있다.
        cap.release()
        out.release()


# ── 4. 통합 전처리 (권장 함수) ─────────────────────────────────────────────────
def process_swing_video_enhanced(input_path, output_path, start_frame, end_frame, crop_box=None, target_fps=30):
    """
    시간 자르기 + 크롭 + FPS 고정 + CLAHE(대비) + Sharpening(샤프닝)을 한 번에 처리한다.

    MediaPipe는 조명이 고르고 경계선이 선명할수록 관절을 더 정확히 인식한다.
    CLAHE와 샤프닝 필터를 적용하여 어두운 영상이나 저화질 영상에서도
    뼈대 추출 품질을 높인다.

    Args:
        input_path  : 원본 영상 경로
        output_path : 전처리된 영상 저장 경로
        start_frame : 스윙 시작 프레임
        end_frame   : 스윙 종료 프레임
        crop_box    : (x, y, w, h) 형태의 크롭 좌표. None이면 전체 화면 사용.
        target_fps  : 출력 영상의 목표 FPS (기본 30fps)

    처리 단계:
        1. 크롭  : crop_box 기준으로 선수 영역만 추출
        2. CLAHE : LAB 색공간의 L채널(밝기)에 적용하여 대비를 국소적으로 최적화
                   (전체 히스토그램 평활화와 달리 과도한 밝기 변화를 억제함)
        3. 샤프닝: 라플라시안 기반 커널로 경계선을 강조하여 관절 인식률 향상
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"영상을 열 수 없습니다: {input_path}")
        return

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    if crop_box:
        x, y, w, h = crop_box
    else:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        x, y = 0, 0

    # source_fps가 target_fps보다 높으면 일부 프레임을 건너뛰어 FPS를 맞춘다.
    frame_step = max(1, int(source_fps / target_fps))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (w, h))

    print(f"고급 전처리 중: {start_frame}~{end_frame} 프레임 | {target_fps}FPS | 크롭 {w}x{h}")
    print("영상 화질 개선 필터(CLAHE & Sharpening) 적용 중...")

    # CLAHE 객체: clipLimit=2.0 은 밝기 증폭 한도, tileGridSize는 국소 영역 분할 수
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    # 라플라시안 샤프닝 커널: 중앙 픽셀을 강조하고 주변을 빼서 경계선을 선명하게 만든다.
    sharpen_kernel = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])

    current_idx = start_frame
    while current_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        # frame_step 간격으로만 저장하여 FPS 조정
        if (current_idx - start_frame) % frame_step == 0:
            # 1. 크롭
            if crop_box:
                frame = frame[y:y+h, x:x+w]

            # 2. CLAHE (대비 최적화)
            # BGR -> LAB 변환 후 L채널(밝기)에만 CLAHE 적용 -> 색상은 보존
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

            # 3. 샤프닝
            frame = cv2.filter2D(frame, -1, sharpen_kernel)

            # 4. 저장
            out.write(frame)

        current_idx += 1

    cap.release()
    out.release()
    print(f"고급 전처리 완료: {output_path}")


# ── 실행 예시 ──────────────────────────────────────────────────────────────────
# 1. 다운로드 (필요시)
# download_video("URL", "data/raw/tiger_raw.mp4")

# 2. 통합 전처리 실행 (예: 타이거 우즈 0~25초, 정면 320x360 크롭, 30FPS 고정)
# tiger_crop = (0, 0, 320, 360)  # 640x360 영상의 왼쪽 절반
# process_swing_video_enhanced("data/raw/tiger_raw.mp4", "data/processed/tiger_final.mp4",
#                              start_frame=0, end_frame=750, crop_box=tiger_crop)

if __name__ == "__main__":
    input_file  = "data/raw/tiger_raw.mp4"
    output_file = "data/processed/tiger_final_enhanced.mp4"

    # 원본(640x360)의 왼쪽 절반(정면 뷰)을 크롭
    tiger_crop = (0, 0, 320, 360)

    print("타이거 우즈 영상 전처리 시작...")

    process_swing_video_enhanced(
        input_path=input_file,
        output_path=output_file,
        start_frame=0,
        end_frame=750,
        crop_box=tiger_crop,
        target_fps=30
    )
