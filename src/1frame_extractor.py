import cv2
import mediapipe as mp
import os

# 1. MediaPipe Pose 솔루션 초기화 (두뇌 가져오기)
mp_pose = mp.solutions.pose
# static_image_mode=True: 비디오가 아니라 '사진 1장' 모드로 작동하게 설정합니다.
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
# 관절을 화면에 예쁘게 그려주는 도구 가져오기
mp_drawing = mp.solutions.drawing_utils 

# 2. 연습용 사진 불러오기
# (파일 이름을 data/raw/ 폴더에 넣은 진짜 사진 이름으로 바꿔주세요!)
image_name = 'practice.jpg' 
image_path = os.path.join('data', 'raw', image_name)

image = cv2.imread(image_path)

# 사진이 제대로 안 열렸을 때 에러 처리
if image is None:
    print(f"🚨 에러: '{image_path}' 경로에서 사진을 찾을 수 없습니다. 파일명을 확인해 주세요.")
    exit()

# 3. 중요: 이미지 전처리 (BGR ➡️ RGB)
# OpenCV는 색상을 BGR 순서로 읽지만, MediaPipe는 RGB 순서로 받습니다. 
# 이 변환을 안 해주면 인식률이 뚝 떨어집니다.
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 4. 🔥 인공지능 실행 (좌표 추출)
# 'RGB 이미지'를 집어넣으면, 인공지능이 계산한 결과(results)를 반환합니다.
results = pose.process(image_rgb)

# 5. 결과물 출력 및 그리기
# 결과를 담을 사진 한 장 복사 (원본 훼손 방지)
annotated_image = image.copy()
h, w, c = image.shape # 사진의 세로, 가로, 채널(색상) 크기 가져오기

# 만약 뼈대 탐지에 성공했다면 (results.pose_landmarks가 존재한다면)
if results.pose_landmarks:
    print(f"🎉 '{image_name}' 사진에서 포즈 탐지에 성공했습니다!")
    
    # 33개 관절 좌표를 하나씩 꺼내서 확인하기
    # (id: 0~32번 관절 번호, lm: x, y, z, visibility가 담긴 좌표 데이터)
    for id, lm in enumerate(results.pose_landmarks.landmark):
        # MediaPipe가 주는 x, y, z는 0.0 ~ 1.0 사이의 '비율(Normalized)' 값입니다.
        # 실제 사진의 픽셀 좌표(px, py)로 바꾸려면 가로(w), 세로(h) 크기를 곱해줘야 합니다.
        px, py = int(lm.x * w), int(lm.y * h)
        
        # 33개 다 찍으면 정신없으니, 0번(코)과 12번(오른쪽 어깨)만 프린트해볼게요.
        if id == 0 or id == 12:
            print(f"📍 관절 ID {id:02} | 비율좌표(x, y): ({lm.x:.4f}, {lm.y:.4f}) | 실제픽셀(px, py): ({px:4}, {py:4}) | 인식률(visibility): {lm.visibility:.2f}")
            
            # (보너스) 실제 픽셀 좌표에 빨간 점 찍고 번호 달기
            cv2.circle(annotated_image, (px, py), 5, (0, 0, 255), -1)
            cv2.putText(annotated_image, str(id), (px+5, py-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # 6. 복사본 사진 위에 뼈대 선(Connections) 예쁘게 그리기
    mp_drawing.draw_landmarks(
        annotated_image, 
        results.pose_landmarks, 
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=4, circle_radius=8), # 관절 점 스타일
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=4, circle_radius=2)  # 뼈대 선 스타일
    )

    # 7. 결과 화면 띄우기
    # ---------------------------------------------------------
    # [새로 추가할 코드 시작] 사진이 너무 커서 화면에 안 보일 때 줄여주는 마법
    # ---------------------------------------------------------
    
    # 1. 원본 사진의 너비(w)와 높이(h) 가져오기
    # (results.pose_landmarks 블록 안에서 사용해야 하므로 블록 밖에서 정의했다면 들여쓰기 확인 필수!)
    h, w, c = image.shape
    print(f"📏 원본 사진 크기: {w}x{h}")

    # 2. 목표로 하는 화면 해상도 설정 (가로를 1000픽셀 정도로 맞춰보겠습니다.)
    target_width = 1000
    
    # 가로 세로 비율을 유지하면서 높이를 자동으로 계산
    scale_factor = target_width / w
    target_height = int(h * scale_factor)

    # 3. 사진 사본(annotated_image)을 이 크기로 줄이기
    resized_image = cv2.resize(annotated_image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    
    print(f"📺 화면에 보여줄 크기: {target_width}x{target_height}")
    # ---------------------------------------------------------
    # [새로 추가할 코드 끝]
    # ---------------------------------------------------------

    # [수정할 부분] 기존 imshow 명령어를 resized_image로 바꿔줍니다.
    # cv2.imshow('MediaPipe Pose Practice (Image)', annotated_image)  <-- 기존
    cv2.imshow('MediaPipe Pose Practice (Image) (Resized)', resized_image) # <-- 수정
    print("\n✅ 창을 끄려면 'q' 키를 누르세요.")
    
    # 키보드 입력을 무한 대기 (아무 키나 누르면 다음으로 넘어감)
    # 사진 한 장이라 waitKey(0)을 씁니다. 30 같은 숫자를 넣으면 그 시간(ms)만큼만 기다려요.
    if cv2.waitKey(0) & 0xFF == ord('q'):
        pass

else:
    print("🚨 포즈 탐지에 실패했습니다. 전신이 나온 다른 사진으로 테스트해 보세요.")

# 8. 메모리 해제
pose.close()
cv2.destroyAllWindows()