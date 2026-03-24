import numpy as np

def calculate_angle(a, b, c):
    """
    세 점의 x, y 좌표를 받아 사이 각도(degrees)를 계산합니다.
    (b점이 각도의 꼭짓점입니다.)
    """
    a = np.array(a) # 첫 번째 점
    b = np.array(b) # 꼭짓점 (예: 팔꿈치나 무릎)
    c = np.array(c) # 세 번째 점

    # 1. 꼭짓점을 기준으로 두 개의 선(벡터)을 만듭니다.
    v1 = a - b
    v2 = c - b

    # 2. 두 선이 이루는 각도를 삼각함수(코사인 제2법칙)로 계산합니다.
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0) # 계산 오차 방지
    
    # 3. 라디안 값을 우리가 아는 '도(degree)'로 바꿉니다.
    angle_rad = np.arccos(cosine_angle)
    angle_deg = np.degrees(angle_rad)

    # 4. 스포츠 분석에서는 보통 180도 이하의 내각을 씁니다.
    if angle_deg > 180.0:
        angle_deg = 360 - angle_deg

    return angle_deg


def normalize_by_pelvis(landmarks):
    """
    33개의 뼈대 좌표를 받아, '골반의 중심'을 (0, 0) 원점으로 영점 조절합니다.
    카메라가 멀리 있든 가까이 있든, 체격이 크든 작든 공평하게 움직임을 비교할 수 있습니다.
    """
    # MediaPipe에서 23번은 왼쪽 엉덩이(골반), 24번은 오른쪽 엉덩이입니다.
    left_hip = landmarks[23]
    right_hip = landmarks[24]
    
    # 1. 두 엉덩이의 정중앙(배꼽 살짝 아래) 좌표를 찾습니다. 여기가 새로운 기준점(0,0)이 됩니다.
    pelvis_x = (left_hip.x + right_hip.x) / 2
    pelvis_y = (left_hip.y + right_hip.y) / 2
    
    normalized_points = []
    
    # 2. 33개의 모든 관절 좌표에서 기준점의 위치만큼 빼줍니다. (영점 조절)
    for lm in landmarks:
        norm_x = lm.x - pelvis_x
        norm_y = lm.y - pelvis_y
        normalized_points.append((norm_x, norm_y))
        
    return normalized_points