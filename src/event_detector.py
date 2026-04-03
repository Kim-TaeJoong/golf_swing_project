import pandas as pd
import numpy as np

def detect_swing_event(df):
    #추가 스무딩
    detect_df = df.rolling(window=5, center=True, min_periods=1).mean()

    events = {}

    #Top of Swing
    top_idx = detect_df['x_factor'].idxmax()
    events['Top of Swing'] = int(df.loc[top_idx, 'frame'])

    #top of swing 이전 부분 검사
    pre_top_df = detect_df.loc[: top_idx]
    #top of swing 이후 부분 검사
    post_top_df = detect_df.loc[top_idx :]

    #Address
    events['Address'] = int(pre_top_df['r_wrist_x'].rolling(window=10).var().idxmin())

    #Takeaway
    address_idx = pre_top_df.index.get_loc(events['address'])
    takeaway_df = pre_top_df.iloc[address_idx:]
    #Address 좌표로 부터 0.05 이동 했을 경우 Takeaway 판단
    if takeaway_mask.any():
        takeaway_mask = takeaway_df['r_wrist_x'] > (takeaway_df['r_wrist_x'].iloc[0] + 0.05)
    else:
        events['takeaway'] = events['address'] + 10

    #Downswing
    downswing_mask = post_top_df['x_factor'].diff() < 0
    if downswing_mask.any():
        # diff() < 0 인 첫 번째 지점이 바로 감소가 시작된 프레임
        events['Downswing'] = int(post_top_df[downswing_mask].index[0])
    else:
        events['Downswing'] = events['top'] + 1

    #Impact
    impact_idx = post_top_df['r_wrist_y'].idxmax()
    events['impact'] = int(df.loc[impact_idx, 'frame'])


    #Finish
    post_impact_df = detect_df.loc[impact_idx:]
    events['finish'] = int(post_impact_df['r_wrist_x'].rolling(window=10).var().idxmin())

    #프레임 순 정렬
    sorted_events = dict(sorted(events.items(), key=lambda item: item[1]))

    return sorted_events