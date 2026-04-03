import pandas as pd
import numpy as np
from utils.utils import calculate_angle, ANGLE_JOINTS, normalize_by_pelvis_csv, calculate_x_factor
import os

csv_path = os.path.join('data','processed','tigerwoods_swing_landmarks_enhanced.csv')
df = pd.read_csv(csv_path)

#신뢰도(v) 낮은 좌표 Nan 처리
for i in range(33):
    df.loc[df[f'v{i}'] < 0.5, [f'x{i}', f'y{i}', f'z{i}']] = np.nan

#좌표 보간
coord_cols = [c for c in df.columns if c.startswith(('x', 'y', 'z'))]
df[coord_cols] = df[coord_cols].interpolate(method='linear', limit_direction='both')

#좌표 스무딩
df[coord_cols] = df[coord_cols].rolling(window=5, min_periods=1, center=True).mean()

'''
#smoothing
cols_to_smooth = [col for col in df.columns if col.startswith(('x', 'y', 'z'))]

# 골라낸 x, y, z 좌표들만 부드럽게 평균을 냅니다. (v 값은 원본 그대로 유지됨)
df[cols_to_smooth] = df[cols_to_smooth].rolling(window=5, min_periods=1, center=True).mean()
'''

results =[]

for idx, row in df.iterrows():
    result = {'frame' : row['frame_number']}

    #골반의 위치 변화 확인
    result['pelvis_raw_x'] = (row['x23'] + row['x24']) / 2
    result['pelvis_raw_z'] = (row['z23'] + row['z24']) / 2

    row = normalize_by_pelvis_csv(row)
    
    #6개의 이벤트 기준을 정하기 위한 손목 좌표
    result['r_wrist_x'] = row['x16']
    result['r_wrist_y'] = row['y16']
    result['l_wrist_x'] = row['y15']

    #머리가 얼마나 흔들리는지 계산
    result['head_norm_x'] = row['x0']
    result['head_norm_y'] = row['y0']

    #양쪽 팔꿈치, 무릎, 손목 각도 계산
    for angle_name, (a, b, c) in ANGLE_JOINTS.items():
        p1 = [row[f'x{a}'], row[f'y{a}'], row[f'z{a}']]
        p2 = [row[f'x{b}'], row[f'y{b}'], row[f'z{b}']]
        p3 = [row[f'x{c}'], row[f'y{c}'], row[f'z{c}']]
        
        #Nan 체크
        if any(np.isnan(v) for v in p1 + p2 + p3):
            result[angle_name] = None
        else:
            result[angle_name] = calculate_angle(p1, p2, p3)
        
        #좌표 보간으로 인해 사용 x
        '''
        if row[f'v{a}'] > 0.5 and row[f'v{b}'] > 0.5 and row[f'v{c}'] > 0.5:
            result[angle_name] = calculate_angle(p1, p2, p3)
        else:
            result[angle_name] = None
        '''
    #척추 각도 계산 (어깨의 중간점과 골반 중간점의 좌표를 계산을 통해 구함)
    #상하체 꼬임 계산

    hip_vals = [row['x11'], row['y11'], row['z11'],
                row['x12'], row['y12'], row['z12'],
                row['x23'], row['y23'], row['z23'],
                row['x24'], row['y24'], row['z24']]
    #Nan 체크
    if any(np.isnan(v) for v in hip_vals):
        result['spine_angle'] = None
        result['x_factor'] = None

    else:
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
    #좌표 보간으로 인한 사용 x
    '''
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
    '''
    #결과 리스트에 누적
    results.append(result)
    
result_df = pd.DataFrame(results)

#각도 좌표에 대한 스무딩 추가(중간값을 사용)
angle_cols = ['r_elbow', 'l_elbow', 'r_shoulder', 'l_shoulder', 'r_knee', 'l_knee', 'r_wrist', 'l_wrist', 'spine_angle', 'x_factor']
result_df[angle_cols] = result_df[angle_cols].rolling(window=3, center=True, min_periods=1).median()
# 2차 평균으로 한번더 제거(과할 수도 있어 일단 사용 x)
#result_df[angle_cols] = result_df[angle_cols].rolling(window=3, min_periods=1, center=True).mean()
result_df.to_csv('data/processed/tigerwoods_angle_enhanced.csv',index= False)