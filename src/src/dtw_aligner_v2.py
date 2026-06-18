import numpy as np
import pandas as pd
import sys
import os

try:
    from dtaidistance import dtw
except ImportError:
    raise ImportError("dtaidistance가 설치되지 않았습니다. 'pip install dtaidistance'를 실행하세요.")

sys.path.insert(0, os.path.dirname(__file__))
from event_detector import detect_swing_event
from utils.utils import (
    calculate_angle, calculate_angle_without_z,
    ANGLE_JOINTS, calculate_x_factor,
)

ANGLE_COLS = [
    'r_elbow', 'l_elbow',
    'r_shoulder', 'l_shoulder',
    'r_knee', 'l_knee',
    'r_wrist', 'l_wrist',
    'spine_angle', 'x_factor',
]

# angle_enhanced.csv 경로에 어깨 x좌표가 없어서
# head_norm_y 절댓값(골반~머리 수직거리)을 체형 스케일 대용으로 사용.
# landmarks 경로에서는 어깨너비 직접 계산 가능하도록 추후 교체 권장.
BODY_SCALE_COL = 'head_norm_y'

POSITION_COLS = ['r_wrist_x', 'r_wrist_y', 'l_wrist_x', 'head_norm_x', 'head_norm_y']


# ──────────────────────────────────────────────
# 내부 유틸
# ──────────────────────────────────────────────

def _load(csv_path: str) -> pd.DataFrame:
    """angle_enhanced.csv 로드 (per-frame 골반 정규화가 이미 적용된 상태)."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['r_wrist_y']).reset_index(drop=True)
    return df


def _get_address_idx(df: pd.DataFrame) -> int:
    """event_detector 로 어드레스 프레임을 찾아 DataFrame 인덱스로 반환."""
    events = detect_swing_event(df)
    address_frame = events.get('Address', df['frame'].iloc[0])
    matches = df.index[df['frame'] == address_frame]
    return int(matches[0]) if len(matches) > 0 else 0


def _process_landmarks(landmarks_path: str) -> pd.DataFrame:
    """
    Raw landmarks CSV -> 어드레스 고정 기준 정규화된 특징 DataFrame.

    기존 analyzer.py 문제(매 프레임 현재 골반을 원점으로 사용) 수정:
    - 어드레스 프레임 골반을 한 번만 계산해 전체 프레임에 고정 적용
    - 골반이 이동해도 머리/손 좌표가 반대로 튀지 않음

    반환 컬럼은 angle_enhanced.csv 와 동일한 구조.
    """
    df = pd.read_csv(landmarks_path).reset_index(drop=True)

    # 1. 신뢰도 낮은 좌표 NaN 처리
    for i in range(33):
        df.loc[df[f'v{i}'] < 0.5, [f'x{i}', f'y{i}', f'z{i}']] = np.nan

    # 2. 선형 보간 + 스무딩
    coord_cols = [c for c in df.columns if c.startswith(('x', 'y', 'z'))]
    df[coord_cols] = df[coord_cols].interpolate(method='linear', limit_direction='both')
    df[coord_cols] = df[coord_cols].rolling(window=5, min_periods=1, center=True).mean()

    # 3. 각도 + per-frame 위치 특징 계산 (어드레스 탐지용)
    #    각도는 원점 무관이므로 이 단계에서 확정.
    #    위치 특징은 어드레스 탐지 후 고정 기준으로 재계산함.
    results = []
    for idx, row in df.iterrows():
        r = {'frame': row['frame_number']}
        r['pelvis_raw_x'] = (row['x23'] + row['x24']) / 2
        r['pelvis_raw_z'] = (row['z23'] + row['z24']) / 2

        # 임시 per-frame 정규화 (어드레스 탐지용으로만 사용)
        px = (row['x23'] + row['x24']) / 2
        py = (row['y23'] + row['y24']) / 2
        r['r_wrist_x']   = row['x16'] - px
        r['r_wrist_y']   = row['y16'] - py
        r['l_wrist_x']   = row['x15'] - px
        r['r_shoulder_y'] = row['y11'] - py
        r['l_hip']       = row['y23'] - py
        r['head_norm_x'] = row['x0']  - px
        r['head_norm_y'] = row['y0']  - py

        # 관절 각도
        for angle_name, (a, b, c) in ANGLE_JOINTS.items():
            p1 = [row[f'x{a}'], row[f'y{a}'], row[f'z{a}']]
            p2 = [row[f'x{b}'], row[f'y{b}'], row[f'z{b}']]
            p3 = [row[f'x{c}'], row[f'y{c}'], row[f'z{c}']]
            if any(np.isnan(v) for v in p1 + p2 + p3):
                r[angle_name] = None
            else:
                r[angle_name] = calculate_angle_without_z(p1, p2, p3)

        # 척추 각도 + X-factor
        hip_vals = [row[f'{ax}{j}'] for ax in ('x', 'y', 'z') for j in (11, 12, 23, 24)]
        if any(np.isnan(v) for v in hip_vals):
            r['spine_angle'] = None
            r['x_factor']   = None
        else:
            sc = [(row['x11']+row['x12'])/2, (row['y11']+row['y12'])/2, (row['z11']+row['z12'])/2]
            hc = [(row['x23']+row['x24'])/2, (row['y23']+row['y24'])/2, (row['z23']+row['z24'])/2]
            vr = [hc[0], hc[1] - 1.0, hc[2]]
            r['spine_angle'] = calculate_angle(sc, hc, vr)
            r['x_factor']    = calculate_x_factor(
                [row['x11'], row['y11'], row['z11']], [row['x12'], row['y12'], row['z12']],
                [row['x23'], row['y23'], row['z23']], [row['x24'], row['y24'], row['z24']],
            )
        results.append(r)

    feat_df = pd.DataFrame(results)

    # 각도 스무딩
    feat_df[ANGLE_COLS] = feat_df[ANGLE_COLS].rolling(window=3, center=True, min_periods=1).median()

    # 4. 어드레스 프레임 탐지
    #    3번 단계의 임시 per-frame 정규화 결과를 이용해 어드레스를 찾는다.
    addr_idx = _get_address_idx(feat_df)
    addr_lm  = df.loc[addr_idx]

    # v2 핵심: 어드레스 시점 골반 중심을 딱 한 번만 계산해 고정 원점으로 사용
    # 기존 방식(매 프레임 현재 골반을 원점으로 사용)에서는
    # 스윙 중 골반이 이동하면 머리·손 좌표가 반대 방향으로 튀는 부작용이 생긴다.
    # 고정 원점을 쓰면 '어드레스 대비 얼마나 움직였는가'를 정확히 추적할 수 있다.
    px_fixed = (addr_lm['x23'] + addr_lm['x24']) / 2
    py_fixed = (addr_lm['y23'] + addr_lm['y24']) / 2

    print(f"[landmarks] address idx: {addr_idx} | fixed pelvis: ({px_fixed:.4f}, {py_fixed:.4f})")

    # 5. 위치 특징을 어드레스 골반 고정 기준으로 재계산
    #    3번 단계에서 구한 임시값을 덮어쓴다.
    feat_df['r_wrist_x']   = df['x16'].values - px_fixed
    feat_df['r_wrist_y']   = df['y16'].values - py_fixed
    feat_df['l_wrist_x']   = df['x15'].values - px_fixed
    feat_df['head_norm_x'] = df['x0'].values  - px_fixed
    feat_df['head_norm_y'] = df['y0'].values  - py_fixed
    feat_df['r_shoulder_y'] = df['y11'].values - py_fixed
    feat_df['l_hip']        = df['y23'].values - py_fixed

    return feat_df, addr_idx


def _normalize(df: pd.DataFrame, addr_idx: int = None) -> tuple:
    """
    어드레스 기준 변화량 정규화.

    좌표 피처: (현재 - 어드레스) / body_scale
    각도 피처: (현재 - 어드레스) / 180
    body_scale = |head_norm_y| at address (골반~머리 수직거리)

    addr_idx: _process_landmarks에서 이미 탐지한 경우 전달해 재탐지를 방지.

    Returns: (normed_df, meta_dict)
    """
    if addr_idx is None:
        addr_idx = _get_address_idx(df)
    addr_row  = df.loc[addr_idx]

    # 체형 스케일: 어드레스 시점의 골반~머리 수직거리.
    # 키가 다른 선수끼리 비교할 때 좌표값의 절대 크기 차이를 제거한다.
    body_scale = abs(addr_row[BODY_SCALE_COL])
    if body_scale < 1e-6:
        body_scale = 1.0  # 데이터 이상치 방어: 0으로 나누기 방지

    normed = {}

    # 위치 피처: 어드레스 기준 상대 변위를 체형 스케일로 나눠 무차원화
    # → 선수마다 다른 카메라 거리·체격 차이를 제거하고 '움직임 비율'만 남긴다
    for col in POSITION_COLS:
        if col in df.columns:
            normed[col] = (df[col] - addr_row[col]) / body_scale

    # 각도 피처: 최대 각도 범위(180°)로 나눠 [−1, 1] 스케일로 통일
    # → 위치 피처와 단위를 맞춰 DTW 거리 계산 시 특정 피처가 과도하게 지배하지 않도록 함
    for col in ANGLE_COLS:
        if col in df.columns:
            normed[col] = (df[col] - addr_row[col]) / 180.0

    meta = {
        'address_idx': addr_idx,
        'body_scale':  body_scale,
    }
    return pd.DataFrame(normed, index=df.index), meta


# ──────────────────────────────────────────────
# 공개 API
# ──────────────────────────────────────────────

def align_swings_v2(
    pro_path: str,
    user_path: str,
    feature: str = 'r_wrist_y',
    window_ratio: float = 0.2,
    use_landmarks: bool = False,
) -> pd.DataFrame:
    """
    개선된 정규화 후 DTW 정렬.

    use_landmarks=True 권장:
        Raw landmarks CSV 에서 어드레스 골반을 고정 기준으로 사용.
        골반 이동시 머리/손 좌표가 반대로 튀는 문제 완전 해결.

    use_landmarks=False (기본):
        기존 angle_enhanced.csv 사용 (per-frame 골반 정규화가 이미 적용된 상태).
        위치 특징의 골반 드리프트 문제가 부분적으로만 보정됨.

    Parameters
    ----------
    pro_path      : 프로 선수 CSV 경로 (landmarks 또는 angle_enhanced)
    user_path     : 비교 대상 CSV 경로
    feature       : DTW 정렬 기준 컬럼 (기본: r_wrist_y)
    window_ratio  : Sakoe-Chiba 밴드 비율 (권장: 0.15~0.25)
    use_landmarks : True 면 raw landmarks CSV 로 처리

    Returns
    -------
    aligned_df 컬럼:
        pro_frame, user_frame,
        pro_{angle}, user_{angle}, diff_{angle}  (ANGLE_COLS 전체)
    """
    # ── Step 1. 데이터 로드 & 위치 특징 재계산 ───────────────────────
    # use_landmarks=True: raw landmarks CSV → 어드레스 골반 고정 기준으로 전처리 (권장)
    # use_landmarks=False: 이미 저장된 angle_enhanced.csv 그대로 사용
    if use_landmarks:
        pro_df,  pro_addr_idx  = _process_landmarks(pro_path)
        user_df, user_addr_idx = _process_landmarks(user_path)
    else:
        pro_df,  pro_addr_idx  = _load(pro_path), None
        user_df, user_addr_idx = _load(user_path), None

    # ── Step 2. 어드레스 기준 정규화 ─────────────────────────────────
    # 두 선수의 좌표·각도를 동일한 스케일로 변환해 DTW 거리 계산에 입력한다.
    pro_normed,  pro_meta  = _normalize(pro_df,  pro_addr_idx)
    user_normed, user_meta = _normalize(user_df, user_addr_idx)

    print(f"[DTW v2] pro  - address idx: {pro_meta['address_idx']}, body_scale: {pro_meta['body_scale']:.4f}")
    print(f"[DTW v2] user - address idx: {user_meta['address_idx']}, body_scale: {user_meta['body_scale']:.4f}")

    # ── Step 3. DTW 정렬 ─────────────────────────────────────────────
    # DTW는 1D 시계열 하나를 기준으로 정렬 경로를 계산한다.
    # 기준 피처(feature)의 정규화된 값을 float64 배열로 변환.
    pro_seq  = pro_normed[feature].fillna(0).values.astype(np.float64)
    user_seq = user_normed[feature].fillna(0).values.astype(np.float64)

    # Sakoe-Chiba 밴드: 두 시퀀스 길이 중 긴 쪽의 window_ratio 비율만큼만
    # 대각선에서 벗어나도록 허용한다.
    # 너무 넓으면 의미 없는 정렬이 생기고, 너무 좁으면 유연성이 사라진다.
    window = int(max(len(pro_seq), len(user_seq)) * window_ratio)
    print(f"[DTW v2] pro frames: {len(pro_seq)}, user frames: {len(user_seq)}, window: {window}")

    # warping_path: 최소 누적 비용 경로를 반환.
    # path = [(pro_idx_0, user_idx_0), (pro_idx_1, user_idx_1), ...]
    # 각 쌍은 "프로의 이 프레임 ↔ 사용자의 이 프레임이 같은 동작 단계"임을 의미한다.
    path = dtw.warping_path(pro_seq, user_seq, window=window)
    print(f"[DTW v2] warping path length: {len(path)}")

    # ── Step 4. 결과 DataFrame 구성 ───────────────────────────────────
    rows = []
    for pro_idx, user_idx in path:
        row = {
            'pro_frame':  int(pro_df.loc[pro_idx,  'frame']),
            'user_frame': int(user_df.loc[user_idx, 'frame']),
        }
        # 정규화 전 원본 각도값을 꺼낸다: feedback에서 실제 각도 차이를 보여줘야 하므로.
        for col in ANGLE_COLS:
            pv = pro_df.loc[pro_idx,  col] if col in pro_df.columns  else np.nan
            uv = user_df.loc[user_idx, col] if col in user_df.columns else np.nan
            row[f'pro_{col}']  = pv
            row[f'user_{col}'] = uv
            # diff: 프로 - 사용자. 양수면 사용자가 프로보다 각도가 작다는 뜻.
            row[f'diff_{col}'] = (
                pv - uv if not (np.isnan(pv) or np.isnan(uv)) else np.nan
            )
        rows.append(row)

    return pd.DataFrame(rows)


def plot_alignment_v2(
    pro_path: str,
    user_path: str,
    feature: str = 'r_wrist_y',
    window_ratio: float = 0.2,
    events: dict = None,
    use_landmarks: bool = False,
    save_path: str = None,
):
    """DTW v2 정렬 전/후 비교 시각화. save_path 지정 시 PNG 저장."""
    import matplotlib.pyplot as plt

    # align_swings_v2와 동일한 Step 1~3을 거쳐 정렬 경로를 구한다.
    if use_landmarks:
        pro_df,  pro_addr_idx  = _process_landmarks(pro_path)
        user_df, user_addr_idx = _process_landmarks(user_path)
    else:
        pro_df,  pro_addr_idx  = _load(pro_path), None
        user_df, user_addr_idx = _load(user_path), None

    pro_normed,  _ = _normalize(pro_df,  pro_addr_idx)
    user_normed, _ = _normalize(user_df, user_addr_idx)

    pro_seq  = pro_normed[feature].fillna(0).values.astype(np.float64)
    user_seq = user_normed[feature].fillna(0).values.astype(np.float64)

    window = int(max(len(pro_seq), len(user_seq)) * window_ratio)
    path   = dtw.warping_path(pro_seq, user_seq, window=window)

    # 워핑 경로를 따라 두 시퀀스를 같은 길이로 늘린다 → 정렬 후 그래프용
    pro_aligned  = np.array([pro_seq[i]  for i, _ in path])
    user_aligned = np.array([user_seq[j] for _, j in path])
    dist = dtw.distance(pro_seq, user_seq, window=window)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    ax1 = axes[0]
    ax1.plot(pro_seq,  label='Pro',  color='tab:blue',   linewidth=1.5)
    ax1.plot(user_seq, label='User', color='tab:orange', linewidth=1.5, alpha=0.8)
    ax1.set_title(f'Before DTW v2 - {feature} (address-normalized)')
    ax1.set_xlabel('Original Frame Index')
    ax1.set_ylabel('Normalized Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(pro_aligned,  label='Pro',              color='tab:blue',   linewidth=1.5)
    ax2.plot(user_aligned, label='User (DTW aligned)', color='tab:orange', linewidth=1.5, alpha=0.8)
    ax2.set_title(f'After DTW v2 - {feature} | DTW dist: {dist:.4f}')
    ax2.set_xlabel('DTW Aligned Index')
    ax2.set_ylabel('Normalized Value')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    if events:
        frame_to_path_idx = {}
        for path_idx, (pro_idx, _) in enumerate(path):
            f = int(pro_df.loc[pro_idx, 'frame'])
            if f not in frame_to_path_idx:
                frame_to_path_idx[f] = path_idx

        cmap = plt.get_cmap('tab10')
        for i, (name, frame_val) in enumerate(events.items()):
            color = cmap(i % 10)
            try:
                orig_idx = pro_df[pro_df['frame'] == frame_val].index[0]
                ax1.axvline(x=orig_idx, color=color, linestyle=':', linewidth=1.5)
                ax1.text(orig_idx, ax1.get_ylim()[1] * 0.95, name,
                         rotation=45, color=color, fontsize=8, fontweight='bold')
            except IndexError:
                pass
            if frame_val in frame_to_path_idx:
                pidx = frame_to_path_idx[frame_val]
                ax2.axvline(x=pidx, color=color, linestyle=':', linewidth=1.5)
                ax2.text(pidx, ax2.get_ylim()[1] * 0.95, name,
                         rotation=45, color=color, fontsize=8, fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[plot] saved: {save_path}")
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    PRO_LM  = os.path.join(ROOT, 'data', 'processed', 'tigerwoods_swing_landmarks_enhanced.csv')
    USER_LM = os.path.join(ROOT, 'data', 'processed', 'mcilroy_swing_landmarks_enhanced.csv')
    OUTPUT  = os.path.join(ROOT, 'data', 'processed', 'dtw_aligned_v2.csv')

    aligned_df = align_swings_v2(PRO_LM, USER_LM, feature='r_wrist_y', window_ratio=0.2, use_landmarks=True)
    aligned_df.to_csv(OUTPUT, index=False)

    print(f"\nSaved: {OUTPUT}")
    print(f"Rows : {len(aligned_df)}")
    print("\n--- preview ---")
    print(aligned_df[['pro_frame', 'user_frame', 'pro_r_elbow', 'user_r_elbow', 'diff_r_elbow']].head(10).to_string(index=False))

    PLOT_OUT = os.path.join(ROOT, 'data', 'processed', 'dtw_alignment_v2.png')
    events = {'Address': 31, 'Takeaway' : 74 ,'Mid-Backswing' : 266, 'Top': 371,'Downswing' : 387,'Impact': 489,'Follow-through' : 536,'Finish' : 735}
    plot_alignment_v2(PRO_LM, USER_LM, feature='r_wrist_y', events=events,
                      use_landmarks=True, save_path=PLOT_OUT)
