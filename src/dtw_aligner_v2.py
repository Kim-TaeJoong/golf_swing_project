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

ANGLE_COLS = [
    'r_elbow', 'l_elbow',
    'r_shoulder', 'l_shoulder',
    'r_knee', 'l_knee',
    'r_wrist', 'l_wrist',
    'spine_angle', 'x_factor',
]

# angle_enhanced.csv에 어깨 x좌표(r_shoulder_x, l_shoulder_x)가 없어서
# head_norm_y의 절댓값(골반~머리 수직 거리)을 체형 스케일 대용으로 사용.
# 추후 analyzer.py에서 어깨 x좌표를 저장하면 shoulder_width로 교체 권장.
BODY_SCALE_COL = 'head_norm_y'

POSITION_COLS = ['r_wrist_x', 'r_wrist_y', 'l_wrist_x', 'head_norm_x', 'head_norm_y']


def _load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['r_wrist_y']).reset_index(drop=True)
    return df


def _get_address_idx(df: pd.DataFrame) -> int:
    """event_detector로 어드레스 프레임을 찾아 DataFrame 인덱스로 반환."""
    events = detect_swing_event(df)
    address_frame = events.get('Address', df['frame'].iloc[0])
    matches = df.index[df['frame'] == address_frame]
    return int(matches[0]) if len(matches) > 0 else 0


def _normalize(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    어드레스 기준 정규화.

    좌표 피처 : (현재값 - 어드레스값) / body_scale
    각도 피처 : (현재값 - 어드레스값) / 180
    body_scale = |head_norm_y| at address (골반~머리 수직거리)

    Returns
    -------
    normed_df : 정규화된 DataFrame (ANGLE_COLS + POSITION_COLS)
    meta      : {'address_idx', 'body_scale', 'address_values'}
    """
    addr_idx = _get_address_idx(df)
    addr_row = df.loc[addr_idx]

    body_scale = abs(addr_row[BODY_SCALE_COL])
    if body_scale < 1e-6:
        body_scale = 1.0

    normed = {}

    for col in POSITION_COLS:
        if col in df.columns:
            normed[col] = (df[col] - addr_row[col]) / body_scale

    for col in ANGLE_COLS:
        if col in df.columns:
            normed[col] = (df[col] - addr_row[col]) / 180.0

    meta = {
        'address_idx': addr_idx,
        'body_scale': body_scale,
        'address_values': addr_row.to_dict(),
    }

    return pd.DataFrame(normed, index=df.index), meta


def align_swings_v2(
    pro_path: str,
    user_path: str,
    feature: str = 'r_wrist_y',
    window_ratio: float = 0.2,
) -> pd.DataFrame:
    """
    어드레스 기준 정규화 + DTW 정렬.

    v1 대비 개선점
    --------------
    - 기준점: 매 프레임 현재 골반 → 어드레스 프레임 고정
      (골반 이동 시 머리/손 좌표가 반대로 튀는 문제 해결)
    - 체형 스케일: 없음 → body_scale(골반~머리 거리)로 나눔
      (두 선수 체격 차이 보정)
    - 피처 스케일 통일: 각도(0~180도) → /180으로 0~1 범위로 맞춤
      (DTW가 각도에만 집중하는 현상 방지)

    Parameters
    ----------
    pro_path     : 프로 선수 angle_enhanced.csv 경로
    user_path    : 유저(또는 다른 프로) angle_enhanced.csv 경로
    feature      : DTW 정렬 기준 컬럼 (기본: r_wrist_y)
    window_ratio : Sakoe-Chiba 밴드 비율 (권장: 0.15~0.25)

    Returns
    -------
    aligned_df 컬럼:
        pro_frame, user_frame,
        pro_{angle}, user_{angle}, diff_{angle}  (ANGLE_COLS 전체)
    """
    pro_df = _load(pro_path)
    user_df = _load(user_path)

    pro_normed, pro_meta = _normalize(pro_df)
    user_normed, user_meta = _normalize(user_df)

    print(f"[DTW v2] 프로  — 어드레스 idx: {pro_meta['address_idx']}, body_scale: {pro_meta['body_scale']:.4f}")
    print(f"[DTW v2] 유저  — 어드레스 idx: {user_meta['address_idx']}, body_scale: {user_meta['body_scale']:.4f}")

    pro_seq = pro_normed[feature].fillna(0).values.astype(np.float64)
    user_seq = user_normed[feature].fillna(0).values.astype(np.float64)

    window = int(max(len(pro_seq), len(user_seq)) * window_ratio)
    print(f"[DTW v2] 프로 프레임: {len(pro_seq)}, 유저 프레임: {len(user_seq)}, 윈도우: {window}")

    path = dtw.warping_path(pro_seq, user_seq, window=window)
    print(f"[DTW v2] 워핑 경로 길이: {len(path)}")

    rows = []
    for pro_idx, user_idx in path:
        row = {
            'pro_frame': int(pro_df.loc[pro_idx, 'frame']),
            'user_frame': int(user_df.loc[user_idx, 'frame']),
        }
        for col in ANGLE_COLS:
            pro_val = pro_df.loc[pro_idx, col] if col in pro_df.columns else np.nan
            user_val = user_df.loc[user_idx, col] if col in user_df.columns else np.nan
            row[f'pro_{col}'] = pro_val
            row[f'user_{col}'] = user_val
            row[f'diff_{col}'] = (
                pro_val - user_val
                if not (np.isnan(pro_val) or np.isnan(user_val))
                else np.nan
            )
        rows.append(row)

    return pd.DataFrame(rows)


def plot_alignment_v2(
    pro_path: str,
    user_path: str,
    feature: str = 'r_wrist_y',
    window_ratio: float = 0.2,
    events: dict = None,
):
    """
    v2 정규화 기준으로 DTW 정렬 전/후 비교 시각화.
    events: {'Address': 53, 'Top': 365, ...} 형태로 넘기면 세로선 표시
    """
    import matplotlib.pyplot as plt

    pro_df = _load(pro_path)
    user_df = _load(user_path)

    pro_normed, _ = _normalize(pro_df)
    user_normed, _ = _normalize(user_df)

    pro_seq = pro_normed[feature].fillna(0).values.astype(np.float64)
    user_seq = user_normed[feature].fillna(0).values.astype(np.float64)

    window = int(max(len(pro_seq), len(user_seq)) * window_ratio)
    path = dtw.warping_path(pro_seq, user_seq, window=window)

    pro_aligned = np.array([pro_seq[i] for i, _ in path])
    user_aligned = np.array([user_seq[j] for _, j in path])
    dist = dtw.distance(pro_seq, user_seq, window=window)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    ax1 = axes[0]
    ax1.plot(pro_seq, label='Pro', color='tab:blue', linewidth=1.5)
    ax1.plot(user_seq, label='User', color='tab:orange', linewidth=1.5, alpha=0.8)
    ax1.set_title(f'Before DTW v2 — {feature}  (address-normalized)')
    ax1.set_xlabel('Original Frame Index')
    ax1.set_ylabel('Normalized Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(pro_aligned, label='Pro', color='tab:blue', linewidth=1.5)
    ax2.plot(user_aligned, label='User (DTW aligned)', color='tab:orange', linewidth=1.5, alpha=0.8)
    ax2.set_title(f'After DTW v2 — {feature}  | DTW dist: {dist:.4f}')
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
    plt.show()


if __name__ == "__main__":
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PRO_PATH = os.path.join(ROOT, 'data', 'processed', 'tigerwoods_angle_enhanced.csv')
    USER_PATH = os.path.join(ROOT, 'data', 'processed', 'mcilroy_angle_enhanced.csv')
    OUTPUT_PATH = os.path.join(ROOT, 'data', 'processed', 'dtw_aligned_v2.csv')

    aligned_df = align_swings_v2(PRO_PATH, USER_PATH, feature='r_wrist_y', window_ratio=0.2)
    aligned_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n정렬 결과 저장: {OUTPUT_PATH}")
    print(f"정렬된 행 수: {len(aligned_df)}")
    print("\n--- 앞부분 미리보기 ---")
    print(aligned_df[['pro_frame', 'user_frame', 'pro_r_elbow', 'user_r_elbow', 'diff_r_elbow']].head(10).to_string(index=False))

    #events = {'Address': 53, 'Takeaway': 83, 'Top': 365, 'Impact': 489, 'Finish': 709}
    events = {'Address': 31, 'Takeaway' : 74 ,'Mid-Backswing' : 266, 'Top': 371,'Downswing' : 387,'Impact': 489,'Follow-through' : 536,'Finish' : 735}
    plot_alignment_v2(PRO_PATH, USER_PATH, feature='r_wrist_y', events=events)
