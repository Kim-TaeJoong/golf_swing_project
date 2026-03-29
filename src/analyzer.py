import pandas as pd
from utils.utils import calculate_angle, ANGLE_JOINTS, normalize_by_pelvis_csv, calculate_x_factor
import os

csv_path = os.path.join('data','processed','tigerwoods_swing_landmarks.csv')
df = pd.read_csv(csv_path)
df = df.dropna()

#smoothing
cols_to_smooth = [col for col in df.columns if col.startswith(('x', 'y', 'z'))]

# 골라낸 x, y, z 좌표들만 부드럽게 평균을 냅니다. (v 값은 원본 그대로 유지됨)
df[cols_to_smooth] = df[cols_to_smooth].rolling(window=5, min_periods=1, center=True).mean()


results =[]

for idx, row in df.iterrows():
    result = {'frame' : row['frame_number']}

    #골반의 위치 변화 확인
    result['pelvis_raw_x'] = (row['x23'] + row['x24']) / 2
    result['pelvis_raw_z'] = (row['z23'] + row['z24']) / 2

    row = normalize_by_pelvis_csv(row)

    #머리가 얼마나 흔들리는지 계산
    result['head_norm_x'] = row['x0']
    result['head_norm_y'] = row['y0']

    #양쪽 팔꿈치, 무릎, 손목 각도 계산
    for angle_name, (a, b, c) in ANGLE_JOINTS.items():
        p1 = [row[f'x{a}'], row[f'y{a}'], row[f'z{a}']]
        p2 = [row[f'x{b}'], row[f'y{b}'], row[f'z{b}']]
        p3 = [row[f'x{c}'], row[f'y{c}'], row[f'z{c}']]

        if row[f'v{a}'] > 0.5 and row[f'v{b}'] > 0.5 and row[f'v{c}'] > 0.5:
            result[angle_name] = calculate_angle(p1, p2, p3)
        else:
            result[angle_name] = None
    #척추 각도 계산 (어깨의 중간점과 골반 중간점의 좌표를 계산을 통해 구함)
    #상하체 꼬임 계산
    if row['v11'] > 0.5 and row['v12'] > 0.5 and row['v23'] > 0.5 and row['v24'] > 0.5:
        shoulder_c = [
            (row['x11'] + row['x12']) / 2,
            (row['y11'] + row['y12']) / 2,
            (row['z11'] + row['z12']) / 2
        ]
        hip_c = [
            (row['x23'] + row['x24']) / 2,
            (row['y23'] + row['y24']) / 2,
            (row['z23'] + row['z24']) / 2
        ]
        #지면 수직선(골반 중심에서 위로 똑바로 올라간 점)
        vertical_ref = [hip_c[0], hip_c[1] - 1.0, hip_c[2]]

        result['spine_angle'] = calculate_angle(shoulder_c, hip_c, vertical_ref)
        
        s_left = [row['x11'], row['y11'], row['z11']]
        s_right = [row['x12'], row['y12'], row['z12']]
        h_left = [row['x23'], row['y23'], row['z23']]
        h_right = [row['x24'], row['y24'], row['z24']]

        result['x_factor'] = calculate_x_factor(s_left, s_right, h_left, h_right)
    else:
        result['spine_angle'] = None
        result['x_factor'] = None

    #결과 리스트에 누적
    results.append(result)
    
result_df = pd.DataFrame(results)
result_df.to_csv('data/processed/tigerwoods_angle.csv',index= False)