import pandas as pd
import numpy as np

def detect_swing_event(df):
    #추가 스무딩
    detect_df = df.rolling(window=5, center=True, min_periods=1).mean()

    events = {}

    #Top of Swing
    threshold = detect_df['x_factor'].mean()
    valid_mask = detect_df['x_factor'] > threshold
    valid_df = detect_df[valid_mask]
    top_idx = valid_df['r_wrist_y'].idxmin()
    events['Top of Swing'] = int(df.loc[top_idx, 'frame'])

    physical_top_idx = detect_df['r_wrist_y'].idxmin()
    
    #top of swing 이전 부분 검사
    pre_top_df = detect_df.loc[: top_idx]
    #top of swing 이후 부분 검사
    post_top_df = detect_df.loc[top_idx :]

    #Address
    address_idx = pre_top_df['r_wrist_x'].rolling(window=10).var().idxmin()
    events['Address'] = int(df.loc[address_idx, 'frame'])

    #Takeaway
    address_idx = pre_top_df.index.get_loc(events['Address'])
    takeaway_df = pre_top_df.iloc[address_idx:]

    start_learn = max(0, address_idx - 20)
    end_learn = min(len(pre_top_df), address_idx + 20)
    address_baseline_data = pre_top_df['r_wrist_x'].iloc[start_learn:end_learn]

    base_mean = address_baseline_data.mean()
    base_std = address_baseline_data.std()
    #동적 임계값(평소 흔들림보다 4배 이상일 경우 Takeaway로 봄)
    dynamic_threshold = max(base_std * 4, 0.005)

    takeaway_mask = (takeaway_df['r_wrist_x'] - base_mean).abs() > dynamic_threshold

    if takeaway_mask.any():
        events['Takeaway'] = int(takeaway_df[takeaway_mask].index[0])
    else:
        events['Takeaway'] = events['Address'] + 10

    #Downswing
    downswing_mask = post_top_df['x_factor'].diff() < 0
    if downswing_mask.any():
        # diff() < 0 인 첫 번째 지점이 바로 감소가 시작된 프레임
        events['Downswing'] = int(post_top_df[downswing_mask].index[0])
    else:
        events['Downswing'] = events['Top of Swing'] + 1


    #Impact
    '''
    #r_wrist_y 위치가 상위 30% 선택
    wrist_threshold = post_top_df['r_wrist_y'].quantile(0.7)

    impact_candidates = post_top_df[
        post_top_df['r_wrist_y'] >= wrist_threshold
    ]
    impact_idx = impact_candidates['x_factor'].idxmin()
    events['Impact'] = int(df.loc[impact_idx, 'frame'])
    '''
    
    # 1. 탑 이후 구간에서 손목이 가장 낮게 내려온 지점(Y값 최대)
    lowest_wrist_idx = post_top_df['r_wrist_y'].idxmax()

    # 2. 손목의 X축 이동 속도(변화량)가 가장 빨라지는 지점을 찾습니다.
    # 타격 직전이 보통 클럽과 손목의 가속이 최대가 되는 지점입니다.
    post_top_df['wrist_v_x'] = post_top_df['r_wrist_x'].diff().abs()

    # 3. 손목 최하점 근처(전후 10프레임 내외)에서 속도가 가장 빠른 지점을 임팩트로 확정합니다.
    search_range = post_top_df.loc[lowest_wrist_idx - 5 : lowest_wrist_idx + 5]
    if not search_range.empty:
        impact_idx = search_range['wrist_v_x'].idxmax()
    else:
        impact_idx = lowest_wrist_idx

    events['Impact'] = int(df.loc[impact_idx, 'frame'])

    #Finish
    post_impact_df = detect_df.loc[impact_idx:]
    events['Finish'] = int(post_impact_df['r_wrist_x'].rolling(window=10).var().idxmin())

    #Mid-Backswing
    backswing_zone = detect_df.loc[events['Takeaway'] : events['Top of Swing']]
    mid_back_idx = (backswing_zone['r_wrist_y'] - backswing_zone['r_shoulder_y']).abs().idxmin()
    events['Mid-Backswing'] = int(df.loc[mid_back_idx, 'frame'])

    #FollowThrough
    
    follow_zone = detect_df.loc[events['Impact'] : events['Finish']]
    lower_condition = follow_zone['r_wrist_y'] > follow_zone['r_shoulder_y']
    valid_zone = follow_zone[lower_condition]
    follow_through_idx = valid_zone['l_elbow'].idxmax()
    events['Follow-through'] = int(df.loc[follow_through_idx, 'frame'])
    

    #프레임 순 정렬
    sorted_events = dict(sorted(events.items(), key=lambda item: item[1]))

    return sorted_events