"""
keypoint_extractor.py — 파이프라인 2단계: 뼈대 좌표 추출

역할:
  전처리된 골프 스윙 영상에서 MediaPipe Pose를 사용해
  매 프레임마다 33개 신체 관절(keypoint)의 x·y·z 좌표와 인식 신뢰도(visibility)를
  CSV 파일로 저장하고, 동시에 뼈대가 그려진 결과 영상도 함께 생성한다.

출력:
  - tigerwoods_swing_landmarks_enhanced.csv : 관절 좌표 데이터 (후속 분석에 사용)
  - tiger_enhanced_with_skeleton.mp4        : 뼈대 오버레이 영상 (시각적 검증용)

MediaPipe 관절 번호 참고:
  11=왼어깨, 12=오른어깨, 13=왼팔꿈치, 14=오른팔꿈치,
  15=왼손목, 16=오른손목, 23=왼골반, 24=오른골반, ...
  (전체 목록: https://google.github.io/mediapipe/solutions/pose.html)
"""

import cv2
import mediapipe as mp
import os
import csv


# ── MediaPipe 초기화 ───────────────────────────────────────────────────────────
# static_image_mode=False: 영상 모드(프레임 간 연속성 추적) → 이미지 모드보다 빠름
# min_detection_confidence: 처음 관절을 탐지할 때 최소 신뢰도 (0.5 = 50%)
# min_tracking_confidence : 이미 탐지된 관절을 추적할 때 최소 신뢰도
mp_pose    = mp.solutions.pose
pose       = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


# ── 입출력 경로 설정 ───────────────────────────────────────────────────────────
video_name = 'tiger_final_enhanced.mp4'  # data/processed/ 폴더에 있는 전처리 완료 영상
video_path = os.path.join('data', 'processed', video_name)

csv_name  = 'tigerwoods_swing_landmarks_enhanced.csv'
csv_path  = os.path.join('data', 'processed', csv_name)

# 뼈대가 그려진 최종 영상 저장 경로 (시각적 검증용)
output_video_path = os.path.join('data', 'processed', 'tiger_enhanced_with_skeleton.mp4')


# ── 영상 열기 ──────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"에러: '{video_path}' 영상을 불러올 수 없습니다.")
    exit()

# 원본 FPS를 그대로 사용하여 결과 영상의 재생 속도가 원본과 동일하게 유지되도록 한다.
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30  # FPS 정보가 없는 영상에 대한 안전 기본값


# ── CSV 헤더 생성 ──────────────────────────────────────────────────────────────
# 컬럼 구조: frame_number, x0, y0, z0, v0, x1, y1, z1, v1, ..., x32, y32, z32, v32
# 총 1 + 33 * 4 = 133개 컬럼
landmarks_header = ['frame_number']
for i in range(33):
    landmarks_header.extend([f'x{i}', f'y{i}', f'z{i}', f'v{i}'])

# ── 메인 처리 루프 ─────────────────────────────────────────────────────────────
with open(csv_path, mode='w', newline='') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(landmarks_header)  # 첫 줄에 헤더 작성

    frame_count = 0
    out = None  # VideoWriter는 첫 프레임 크기 확인 후 초기화

    print(f"비디오 분석 시작! 데이터는 '{csv_path}'에 실시간으로 저장됩니다...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"분석 완료! 좌표 데이터가 '{csv_path}'에 성공적으로 저장되었습니다.")
            break

        frame_count += 1

        # 모니터 화면에 맞게 리사이즈 (원본 비율 유지)
        h, w, c = frame.shape
        target_width  = 600
        scale_factor  = target_width / w
        target_height = int(h * scale_factor)
        frame_resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

        # 첫 프레임에서 리사이즈된 크기를 확인한 뒤 VideoWriter를 초기화한다.
        if out is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4v 코덱 (macOS/Windows 호환)
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (target_width, target_height))

        # MediaPipe는 BGR이 아닌 RGB 이미지를 입력받는다.
        image_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        results   = pose.process(image_rgb)

        if results.pose_landmarks:
            # 33개 관절 좌표를 한 행으로 평탄화하여 CSV에 저장
            frame_data = [frame_count]
            for landmark in results.pose_landmarks.landmark:
                frame_data.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])

            # 뼈대 시각화: 관절은 빨간색 원, 연결선은 초록색
            mp_drawing.draw_landmarks(
                frame_resized,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=4, circle_radius=6),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=4, circle_radius=2)
            )
        else:
            # 관절을 인식하지 못한 프레임은 빈 값('')으로 채워 행 수를 유지한다.
            frame_data = [frame_count] + [''] * (33 * 4)

        csv_writer.writerow(frame_data)

        # 뼈대가 그려진 프레임을 결과 영상에 기록
        out.write(frame_resized)

        cv2.imshow('Golf Swing AI Analysis & Data Extraction', frame_resized)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            print("분석 종료.")
            break


# ── 자원 해제 ──────────────────────────────────────────────────────────────────
cap.release()

# VideoWriter를 닫지 않으면 저장된 영상이 재생 불가능한 상태로 남는다.
if out is not None:
    out.release()

pose.close()
cv2.destroyAllWindows()

print(f"분석 완료! 뼈대 영상이 성공적으로 저장되었습니다: {output_video_path}")
