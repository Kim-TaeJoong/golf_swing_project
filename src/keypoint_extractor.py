import cv2
import mediapipe as mp
import os
import csv  # 🔥 CSV 파일을 다루기 위한 기본 라이브러리 추가

# 1. MediaPipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 2. 입출력 파일 경로 설정
video_name = 'my_video_cropped.mp4' # <--- data/raw/ 폴더에 있는 실제 영상 이름
video_path = os.path.join('data', 'raw', video_name)

# 🔥 데이터를 저장할 새로운 경로 설정 (processed 폴더에 저장)
csv_name = 'tigerwoods_swing_landmarks.csv'
csv_path = os.path.join('data', 'processed', csv_name)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"🚨 에러: '{video_path}' 영상을 불러올 수 없습니다.")
    exit()

# 🔥 3. CSV 파일 쓰기 준비 및 헤더(머리글) 만들기
# 엑셀의 첫 번째 줄에 들어갈 이름들을 만듭니다 (예: frame, x0, y0, z0, v0, x1, y1 ...)
landmarks_header = ['frame_number']
for i in range(33):
    # 각 관절(0~32번)마다 x, y, z 좌표와 인식률(v)을 저장합니다.
    landmarks_header.extend([f'x{i}', f'y{i}', f'z{i}', f'v{i}'])

# 파일을 'w'(쓰기) 모드로 열기 (안전하게 with 문 사용)
with open(csv_path, mode='w', newline='') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(landmarks_header) # 첫 줄에 머리글 쓰기

    frame_count = 0 # 현재 몇 번째 프레임인지 세는 변수
    print(f"🎬 비디오 분석 시작! 데이터는 '{csv_path}'에 실시간으로 저장됩니다...")

    # 4. 비디오 프레임 반복 루프
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"✅ 분석 완료! 좌표 데이터가 '{csv_path}'에 성공적으로 저장되었습니다.")
            break

        frame_count += 1
        
        # [크기 조절] 모니터에 한눈에 들어오도록 세팅
        h, w, c = frame.shape
        target_width = 1000
        scale_factor = target_width / w
        target_height = int(h * scale_factor)
        frame_resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

        # 이미지 전처리 및 AI 추론
        image_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        # 🔥 5. 뼈대를 찾았다면 CSV 파일에 좌표 저장하기
        if results.pose_landmarks:
            # 현재 프레임의 모든 데이터를 담을 빈 리스트
            frame_data = [frame_count]
            
            # 33개 관절 데이터를 순서대로 꺼내서 리스트에 이어 붙이기
            for landmark in results.pose_landmarks.landmark:
                frame_data.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])

            # 화면에 뼈대 그리기 (시각화)
            mp_drawing.draw_landmarks(
                frame_resized, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=4, circle_radius=6),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=4, circle_radius=2)
            )
        else:
            frame_data = [frame_count] + [''] * (33 * 4)
        # 엑셀 파일에 한 줄(Row) 추가하기!
        csv_writer.writerow(frame_data)
        # 화면 띄우기
        cv2.imshow('Golf Swing AI Analysis & Data Extraction', frame_resized)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            print("분석 종료.")
            break

# 6. 메모리 해제
cap.release()
pose.close()
cv2.destroyAllWindows()