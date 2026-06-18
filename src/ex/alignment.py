import numpy as np
import pandas as pd
from dtaidistance import dtw
from event_detector import detect_swing_event
import os

def run_milestone_detection(df):
    # 1. 데이터 로드
    '''
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        return None
    '''

    # 2. 이정표 탐색 (이벤트 디텍터 호출)
    # 내부적으로 5프레임 스무딩을 거쳐 안정적인 인덱스를 찾습니다.
    milestones = detect_swing_event(df)
    
    # 3. 결과 출력 및 검증
    print("\n[추출된 스윙 이정표]")
    for event, frame in milestones.items():
        print(f"📍 {event:<10} : {frame:>4} 프레임")
        
    return milestones

def get_phase_percentage_data(df, events, points_per_phase):
    milestone = list(events.keys())
    # 1. 정규화할 컬럼 리스트 (frame 제외)
    target_columns = [col for col in df.columns if col != 'frame']
    #딕셔너리로 초기화
    percentage_data = {col: [] for col in target_columns}
    for i in range (len(milestone)-1):
        start_f = events[milestone[i]]
        end_f = events[milestone[i+1]]

        phase_df = df[df['frame'].between(start_f, end_f)]

        for col in target_columns:
            sagment_values = phase_df[col].values
            if len(sagment_values) > 1:
                # 기존 데이터의 인덱스(0~1)를 새로운 100개의 지점으로 매핑
                x_old = np.linspace(0, 1, len(sagment_values))
                x_new = np.linspace(0, 1, points_per_phase)

                # 선형 보간법을 통한 데이터 재구성
                resampled_data = np.interp(x_new, x_old, sagment_values)
                percentage_data[col].extend(resampled_data)

            else:
                # 구간 내 데이터가 부족할 경우 예외 처리 (동일 값으로 채움)
                fill_value = sagment_values[0] if len(sagment_values) > 0 else 0
                percentage_data[col].extend([fill_value] * points_per_phase)

    return pd.DataFrame(percentage_data)

def calculate_tempo(df, events):
    #이상적 비율은 3:1, 2.8 : 1 ~ 3.2 : 1
    backswing = events['Top of Swing'] - events['Address']
    downswing = events['Impact'] - events['Downswing']

    if downswing > 0 :
        tempo_ratio = backswing/downswing
    else :
        tempo_ratio = 0
    
    return backswing, downswing, tempo_ratio

def pose_score(pro_df, df):
    df_diff = pro_df.values - df.values

    rmse_df = np.sqrt(np.mean(df_diff**2, axis=0))
    total_score = pd.Series(rmse_df, index= pro_df.columns)

    phases = {
        "1_Takeaway (어드레스 ~ 테이크어웨이)": (0, 100),
        "2_Backswing (테이크어웨이 ~ 미드백스윙)": (100, 200),
        "3_Top (미드백스윙 ~ 탑)": (200, 300),
        "4_Downswing (탑 ~ 다운스윙 시작)": (300, 400),
        "5_Impact (다운스윙 ~ 임팩트)": (400, 500),
        "6_FollowThrough (임팩트 ~ 팔로우스루)": (500, 600),
        "7_Finish (팔로우스루 ~ 피니시)": (600, 700)
    }

    phase_rmse_dict = {}

    for phase_name, (start_idx, end_idx) in phases.items():
        pro_phase = pro_df.iloc[start_idx : end_idx].values
        df_phase = df.iloc[start_idx : end_idx].values

        phase_diff = pro_phase - df_phase
        phase_rmse = np.sqrt(np.mean(phase_diff**2, axis= 0))

        phase_rmse_dict[phase_name] = pd.Series(phase_rmse, index = pro_df.columns)


    return total_score, phase_rmse_dict

if __name__ == "__main__":
    # 타이거 우즈의 데이터 경로 (본인의 경로에 맞게 수정하세요)
    print("프로그램이 시작되었습니다!")
    data_path1 = 'data/processed/tigerwoods_angle_enhanced.csv'
    pro_df = pd.read_csv(data_path1)
    data_path2 = 'data/processed/mcilroy_angle_enhanced.csv'
    df = pd.read_csv(data_path2)
    data_path3 = 'data/processed/morikawa_angle_enhanced.csv'
    df3 = pd.read_csv(data_path3)
    mcilroy_milestones = run_milestone_detection(df)
    tiger_milestones = run_milestone_detection(pro_df)
    morikawa_milestones = run_milestone_detection(df3)
    '''pro_percent_data = get_phase_percentage_data(pro_df,tiger_milestones,100)
    mcilroy_percent_data = get_phase_percentage_data(df,mcilroy_milestones,100)
    print("=== 프로의 오른쪽 팔꿈치 각도 ===")
    print(pro_df.loc[365:375, 'r_elbow'])
    print(pro_df.loc[365:375, 'l_elbow'])

    print("\n=== 내 영상의 오른쪽 팔꿈치 각도 ===")
    print(df.loc[290:300, 'r_elbow'])
    print(df.loc[290:300, 'l_elbow'])
    #print(mcilroy_percent_data.head())
    #print(mcilroy_percent_data)
    backswing, downswing, tempo = calculate_tempo(df, mcilroy_milestones)
    print(tempo)
    total_rmse, phase_rmse = pose_score(pro_percent_data, mcilroy_percent_data)
    angle_cols = ['r_elbow', 'l_elbow', 'r_shoulder', 'l_shoulder', 
                'r_knee', 'l_knee', 'r_wrist', 'l_wrist', 'spine_angle', 'x_factor']'''
    # 1. 딕셔너리를 데이터프레임으로 변환 (가로: 스윙 구간, 세로: 관절 이름)
    '''all_phases_df = pd.DataFrame(phase_rmse)

    # 2. 각도 데이터만 추출 (픽셀 좌표와 분리하기 위함)
    angle_cols = ['r_elbow', 'l_elbow', 'r_shoulder', 'l_shoulder', 
                'r_knee', 'l_knee', 'r_wrist', 'l_wrist', 'spine_angle', 'x_factor']

    # 3. 원하는 관절들만 뽑아서 소수점 2자리로 출력 (.to_string()으로 중간 생략 방지)
    print("=== 🏌️ 스윙 전체 흐름: 구간별 각도 오차(RMSE) 종합표 ===")
    print(all_phases_df.loc[angle_cols].round(2).to_string())'''
    print("=== 🔍 구간별 상세 오차 리포트 ===")

# 딕셔너리의 Key(구간명)와 Value(Series)를 하나씩 꺼내며 반복
    '''for phase_name, series_data in phase_rmse.items():
        print(f"\n[{phase_name}]")
        
        # 각도 데이터만 뽑아서 소수점 2자리로 출력
        angles_only = series_data[angle_cols].round(2)
        print(angles_only.to_string())

    print("[total_rmse]")
    print(total_rmse)'''