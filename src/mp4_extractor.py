import cv2
import mediapipe as mp
import os
import csv
from collections import deque
# 🔥 우리가 방금 만든 수학 도구함에서 '각도 계산기'를 가져옵니다!
from utils import calculate_angle

# 1. MediaPipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 스윙 패스 시각화 설정 (오른쪽 손목: 16번)
target_landmark_id = 16 
max_points = 50 
trajectory_pts = deque(maxlen=max_points)

# 2. 파일 경로 설정
video_name = 'posepractice.mp4' 
video_path = os.path.join('data', 'raw', video_name)
csv_name = 'swing_landmarks_advanced.csv'
csv_path = os.path.join('data', 'processed', csv_name)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"🚨 에러: '{video_path}' 영상을 불러올 수 없습니다.")
    exit()

# 3. CSV 파일 준비
landmarks_header = ['frame_number']
for i in range(33):
    landmarks_header.extend([f'x{i}', f'y{i}', f'z{i}', f'v{i}'])

with open(csv_path, mode='w', newline='') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(landmarks_header)

    frame_count = 0 
    print(f"🎬 비디오 분석 & 실시간 팔꿈치 각도 측정 시작!")

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"✅ 분석 완료!")
            break

        frame_count += 1
        
        # 화면 크기 조절
        h, w, c = frame.shape
        target_width = 1000
        scale_factor = target_width / w
        target_height = int(h * scale_factor)
        frame_resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

        # AI 분석
        image_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            # [CSV 데이터 저장]
            frame_data = [frame_count]
            for landmark in results.pose_landmarks.landmark:
                frame_data.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
            csv_writer.writerow(frame_data)

            # [스윙 패스 좌표 수집]
            current_target = results.pose_landmarks.landmark[target_landmark_id]
            if current_target.visibility > 0.7:
                px = int(current_target.x * target_width)
                py = int(current_target.y * target_height)
                trajectory_pts.append((px, py))

            # 뼈대 그리기
            mp_drawing.draw_landmarks(
                frame_resized, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=4, circle_radius=6),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=4, circle_radius=2)
            )

            # -----------------------------------------------------------------
            # 🔥 [핵심 추가 기능] 실시간 오른쪽 팔꿈치 각도 계산 및 화면 표시
            # -----------------------------------------------------------------
            # 오른쪽 어깨(12), 오른쪽 팔꿈치(14), 오른쪽 손목(16) 데이터를 꺼냅니다.
            r_shoulder = results.pose_landmarks.landmark[12]
            r_elbow = results.pose_landmarks.landmark[14]
            r_wrist = results.pose_landmarks.landmark[16]

            # 세 관절 모두 화면에 잘 보일 때만 (신뢰도 0.5 이상) 각도를 계산합니다.
            if r_shoulder.visibility > 0.5 and r_elbow.visibility > 0.5 and r_wrist.visibility > 0.5:
                
                # 💡 중요: 비율 좌표(0~1)를 화면의 진짜 픽셀 좌표로 바꿔서 계산해야 각도가 왜곡되지 않습니다.
                p1 = [int(r_shoulder.x * target_width), int(r_shoulder.y * target_height)]
                p2 = [int(r_elbow.x * target_width), int(r_elbow.y * target_height)] # 꼭짓점
                p3 = [int(r_wrist.x * target_width), int(r_wrist.y * target_height)]

                # utils.py의 calculate_angle 함수 호출!
                elbow_angle = calculate_angle(p1, p2, p3)

                # 화면에 예쁘게 글씨 쓰기 (팔꿈치 좌표 기준 오른쪽 아래에 배치)
                text_x = p2[0] + 15
                text_y = p2[1] + 20
                
                # 글씨가 잘 보이게 검은색 테두리를 먼저 그리고, 그 위에 노란색 글씨를 씁니다.
                cv2.putText(frame_resized, f"{int(elbow_angle)} deg", (text_x, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4) # 테두리
                cv2.putText(frame_resized, f"{int(elbow_angle)} deg", (text_x, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2) # 노란색 글씨

        # [스윙 패스 잔상 그리기]
        for i in range(1, len(trajectory_pts)):
            base_thickness = 15 # ✅ 선의 가장 굵은 기본 두께를 15 픽셀
            # 궤적 리스트 앞쪽(오래된 점)일수록 더 얇게 그립니다.
            thickness = int(base_thickness * (i / float(max_points)))
            if thickness < 1: thickness = 1 # 최소 두께 보장
            
            cv2.line(frame_resized, trajectory_pts[i - 1], trajectory_pts[i], (255, 0, 0), thickness)

        cv2.imshow('Golf Swing Analysis Pro', frame_resized)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            print("🛑 강제 종료")
            break

cap.release()
pose.close()
cv2.destroyAllWindows()