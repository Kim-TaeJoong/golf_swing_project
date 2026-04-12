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
    #r_wrist_y 위치가 상위 30% 선택
    wrist_threshold = post_top_df['r_wrist_y'].quantile(0.7)

    impact_candidates = post_top_df[
        post_top_df['r_wrist_y'] >= wrist_threshold
    ]
    impact_idx = impact_candidates['x_factor'].idxmin()
    events['Impact'] = int(df.loc[impact_idx, 'frame'])

    #Finish
    post_impact_df = detect_df.loc[top_idx:]
    events['Finish'] = int(post_impact_df['r_wrist_x'].rolling(window=10).var().idxmin())

    #Mid-Backswing
    

    #프레임 순 정렬
    sorted_events = dict(sorted(events.items(), key=lambda item: item[1]))

    return sorted_events