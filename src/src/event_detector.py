import pandas as pd
import numpy as np

def detect_swing_event(df):
    # 이벤트 탐지 전 추가 스무딩: 노이즈로 인한 오탐 방지
    detect_df = df.rolling(window=5, center=True, min_periods=1).mean()

    events = {}

    # ── Top of Swing ──────────────────────────────────────────────────
    # 백스윙 정점: X-factor(상하체 꼬임)가 평균 이상인 구간 중에서
    # 오른손목이 가장 높이 올라간(y값 최소) 프레임을 Top으로 정의한다.
    # X-factor 조건을 추가하는 이유: 단순 손목 최고점만 쓰면
    # 셋업 자세(Address)가 잘못 잡힐 수 있기 때문이다.
    threshold = detect_df['x_factor'].mean()
    valid_mask = detect_df['x_factor'] > threshold
    valid_df = detect_df[valid_mask]
    top_idx = valid_df['r_wrist_y'].idxmin()
    events['Top of Swing'] = int(df.loc[top_idx, 'frame'])

    physical_top_idx = detect_df['r_wrist_y'].idxmin()

    # Top 이전 구간 = 백스윙, Top 이후 구간 = 다운스윙 이후
    pre_top_df = detect_df.loc[: top_idx]
    post_top_df = detect_df.loc[top_idx :]

    # ── Address ───────────────────────────────────────────────────────
    # 어드레스: 백스윙 시작 전 정지 상태.
    # 오른손목 x좌표의 분산이 가장 작은 구간 = 손이 가장 흔들리지 않는 순간.
    address_idx = pre_top_df['r_wrist_x'].rolling(window=10).var().idxmin()
    events['Address'] = int(df.loc[address_idx, 'frame'])

    # ── Takeaway ──────────────────────────────────────────────────────
    # 테이크어웨이: 클럽이 셋업 위치를 벗어나 백스윙이 시작되는 순간.
    address_idx = pre_top_df.index.get_loc(events['Address'])
    takeaway_df = pre_top_df.iloc[address_idx:]

    # 어드레스 전후 20프레임을 '정지 기준 구간'으로 삼아 평균·표준편차를 구한다.
    start_learn = max(0, address_idx - 20)
    end_learn = min(len(pre_top_df), address_idx + 20)
    address_baseline_data = pre_top_df['r_wrist_x'].iloc[start_learn:end_learn]

    base_mean = address_baseline_data.mean()
    base_std = address_baseline_data.std()

    # 동적 임계값: 평소 정지 흔들림(base_std)의 4배를 초과하면 의도적인 움직임으로 판단.
    # 최솟값 0.005 는 표준편차가 극히 작을 때 잡음에 과민 반응하지 않도록 보정.
    dynamic_threshold = max(base_std * 4, 0.005)

    takeaway_mask = (takeaway_df['r_wrist_x'] - base_mean).abs() > dynamic_threshold

    if takeaway_mask.any():
        events['Takeaway'] = int(takeaway_df[takeaway_mask].index[0])
    else:
        events['Takeaway'] = events['Address'] + 10

    '''#Downswing
    downswing_mask = post_top_df['x_factor'].diff() < 0
    if downswing_mask.any():
        # diff() < 0 인 첫 번째 지점이 바로 감소가 시작된 프레임
        events['Downswing'] = int(post_top_df[downswing_mask].index[0])
    else:
        events['Downswing'] = events['Top of Swing'] + 1'''

    # Downswing
    # 1. 각 프레임이 이전 프레임보다 감소했는지(True/False) 확인
    is_decreasing = post_top_df['x_factor'].diff() < 0
    
    # 2. 5프레임 단위로 True의 개수를 합산
    rolling_dec_sum = is_decreasing.rolling(window=5).sum()
    
    # 3. 5연속 감소 조건을 만족하는 구간 찾기
    consecutive_mask = rolling_dec_sum == 5
    
    if consecutive_mask.any():
        # 조건을 만족한 첫 번째 지점
        end_idx = post_top_df[consecutive_mask].index[0]
        
        # 시작 프레임(첫 번째 감소 프레임)으로 되돌아가기 위해 인덱스 위치에서 4를 뺌
        end_pos = post_top_df.index.get_loc(end_idx)
        start_idx = post_top_df.index[max(0, end_pos - 4)]
        
        events['Downswing'] = int(df.loc[start_idx, 'frame'])
        
    elif is_decreasing.any():
        # 혹시 영상이 너무 짧거나 데이터가 부족해서 5연속이 안 나오면, 
        # 안전장치(Fallback)로 기존처럼 1프레임 감소라도 잡습니다.
        events['Downswing'] = int(post_top_df[is_decreasing].index[0])
        
    else:
        events['Downswing'] = events['Top of Swing'] + 1

    # ── Impact ────────────────────────────────────────────────────────
    # 임팩트: 클럽이 공과 접촉하는 순간.
    # 두 조건의 교차점으로 찾는다:
    #   ① 손목이 가장 낮게 내려온 지점(지면에 가장 가까운 순간)
    #   ② 그 근방에서 손목의 수평 이동 속도가 가장 빠른 프레임
    # → 둘 다 만족하는 지점이 클럽헤드 가속이 최대인 임팩트 구간과 일치한다.

    # 1. 탑 이후 구간에서 손목 y좌표 최대 지점(화면 좌표계: y↓ 증가 = 가장 낮은 위치)
    lowest_wrist_idx = post_top_df['r_wrist_y'].idxmax()

    # 2. 손목 x 이동 속도(프레임 간 변화량 절댓값)
    post_top_df['wrist_v_x'] = post_top_df['r_wrist_x'].diff().abs()

    # 3. 손목 최하점 ±5프레임 범위 내에서 속도 최대 지점을 임팩트로 확정
    search_range = post_top_df.loc[lowest_wrist_idx - 5 : lowest_wrist_idx + 5]
    if not search_range.empty:
        impact_idx = search_range['wrist_v_x'].idxmax()
    else:
        impact_idx = lowest_wrist_idx

    events['Impact'] = int(df.loc[impact_idx, 'frame'])

    # ── Finish ────────────────────────────────────────────────────────
    # 피니시: 임팩트 이후 손이 다시 정지하는 순간.
    # Address와 동일한 논리: x 분산이 최소인 구간 = 동작이 완결된 정지점.
    post_impact_df = detect_df.loc[impact_idx:]
    events['Finish'] = int(post_impact_df['r_wrist_x'].rolling(window=10).var().idxmin())

    # ── Mid-Backswing ─────────────────────────────────────────────────
    # 미드백스윙: 테이크어웨이~탑 구간 중 손목 높이와 어깨 높이가 가장 가까운 프레임.
    # 이 시점에 클럽이 수평(9시 방향)에 위치한다는 스윙 원리를 근거로 한다.
    backswing_zone = detect_df.loc[events['Takeaway'] : events['Top of Swing']]
    mid_back_idx = (backswing_zone['r_wrist_y'] - backswing_zone['r_shoulder_y']).abs().idxmin()
    events['Mid-Backswing'] = int(df.loc[mid_back_idx, 'frame'])

    # ── Follow-through ────────────────────────────────────────────────
    # 팔로스루: 임팩트~피니시 구간 중 손이 어깨보다 위에 있는 상태에서
    # 왼쪽 팔꿈치가 가장 많이 구부러진 프레임.
    # 왼팔이 접히기 시작하는 시점 = 클럽이 몸을 감아 올라가는 팔로스루의 특징.
    follow_zone = detect_df.loc[events['Impact'] : events['Finish']]
    lower_condition = follow_zone['r_wrist_y'] > follow_zone['r_shoulder_y']
    valid_zone = follow_zone[lower_condition]
    follow_through_idx = valid_zone['l_elbow'].idxmax()
    events['Follow-through'] = int(df.loc[follow_through_idx, 'frame'])
    

    #프레임 순 정렬
    sorted_events = dict(sorted(events.items(), key=lambda item: item[1]))

    return sorted_events