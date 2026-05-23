import numpy as np
import pandas as pd
try:
    from dtaidistance import dtw
except ImportError:
    raise ImportError("dtaidistance가 설치되지 않았습니다. 'pip install dtaidistance' 를 실행하세요.")

ANGLE_COLS = [
    'r_elbow', 'l_elbow',
    'r_shoulder', 'l_shoulder',
    'r_knee', 'l_knee',
    'r_wrist', 'l_wrist',
    'spine_angle', 'x_factor',
]

POSITION_COLS = ['r_wrist_x', 'r_wrist_y', 'l_wrist_x', 'head_norm_x', 'head_norm_y']

ALL_COLS = ANGLE_COLS + POSITION_COLS


def _zscore(series: pd.Series) -> np.ndarray:
    arr = series.values.astype(np.float64)
    mean, std = np.nanmean(arr), np.nanstd(arr)
    return (arr - mean) / (std + 1e-9)


def _load(csv_path: str, feature: str):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[feature]).reset_index(drop=True)
    return df


def align_swings(
    pro_path: str,
    user_path: str,
    feature: str = 'r_wrist_y',
    window_ratio: float = 0.2,
) -> pd.DataFrame:
    """
    DTW 워핑 경로로 프로/유저 스윙을 정렬한 DataFrame을 반환합니다.

    Parameters
    ----------
    pro_path      : 프로 선수 angle_enhanced.csv 경로
    user_path     : 유저 angle_enhanced.csv 경로
    feature       : DTW 정렬 기준 컬럼 (기본: r_wrist_y)
    window_ratio  : Sakoe-Chiba 밴드 비율 — 전체 길이의 몇 % 이내에서만 워핑 허용
                    너무 크면 비현실적 매핑, 너무 작으면 정렬 실패 (권장: 0.15~0.25)

    Returns
    -------
    aligned_df 컬럼 구조:
        pro_frame, user_frame,
        pro_{angle}, user_{angle}, diff_{angle}  (ANGLE_COLS 전체)
    """
    pro_df = _load(pro_path, feature)
    user_df = _load(user_path, feature)

    pro_seq = _zscore(pro_df[feature])
    user_seq = _zscore(user_df[feature])

    # Sakoe-Chiba 밴드: 두 시퀀스 중 긴 것 기준으로 window 크기 결정
    window = int(max(len(pro_seq), len(user_seq)) * window_ratio)

    print(f"[DTW] 프로 프레임: {len(pro_seq)}, 유저 프레임: {len(user_seq)}, 윈도우: {window}")
    path = dtw.warping_path(pro_seq, user_seq, window=window)
    print(f"[DTW] 워핑 경로 길이: {len(path)}")

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
            # diff: 양수면 프로가 더 큰 각도, 음수면 유저가 더 큰 각도
            row[f'diff_{col}'] = pro_val - user_val if not (np.isnan(pro_val) or np.isnan(user_val)) else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def plot_alignment(
    pro_path: str,
    user_path: str,
    feature: str = 'r_wrist_y',
    window_ratio: float = 0.2,
    events: dict = None,
):
    """
    DTW 정렬 전/후 비교 시각화.
    events: {'Address': 53, 'Top': 365, ...} 형태로 넘기면 프로 기준 이벤트 세로선 표시
    """
    import matplotlib.pyplot as plt

    pro_df = _load(pro_path, feature)
    user_df = _load(user_path, feature)

    pro_seq = _zscore(pro_df[feature])
    user_seq = _zscore(user_df[feature])

    window = int(max(len(pro_seq), len(user_seq)) * window_ratio)
    path = dtw.warping_path(pro_seq, user_seq, window=window)

    # 워핑 경로로 정렬된 시퀀스 생성
    pro_aligned = np.array([pro_seq[i] for i, _ in path])
    user_aligned = np.array([user_seq[j] for _, j in path])

    dist = dtw.distance(pro_seq, user_seq, window=window)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    # --- 정렬 전 ---
    ax1 = axes[0]
    ax1.plot(range(len(pro_seq)), pro_seq, label='Pro (Tiger)', color='tab:blue', linewidth=1.5)
    ax1.plot(range(len(user_seq)), user_seq, label='User', color='tab:orange', linewidth=1.5, alpha=0.8)
    ax1.set_title(f'Before DTW — {feature}  (original time axis)')
    ax1.set_xlabel('Original Frame Index')
    ax1.set_ylabel('Z-score')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- 정렬 후 ---
    ax2 = axes[1]
    ax2.plot(pro_aligned, label='Pro (Tiger)', color='tab:blue', linewidth=1.5)
    ax2.plot(user_aligned, label='User (DTW aligned)', color='tab:orange', linewidth=1.5, alpha=0.8)
    ax2.set_title(f'After DTW — {feature}  (warped time axis) | DTW dist: {dist:.4f}')
    ax2.set_xlabel('DTW Aligned Index')
    ax2.set_ylabel('Z-score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 이벤트 세로선 (프로 기준 프레임 → 정렬 인덱스로 변환)
    if events:
        pro_frames = pro_df['frame'].tolist()
        # 각 이벤트 프레임에 해당하는 워핑 경로 인덱스 찾기
        frame_to_path_idx = {}
        for path_idx, (pro_idx, _) in enumerate(path):
            f = int(pro_df.loc[pro_idx, 'frame'])
            if f not in frame_to_path_idx:
                frame_to_path_idx[f] = path_idx

        cmap = plt.get_cmap('tab10')
        for i, (name, frame_val) in enumerate(events.items()):
            color = cmap(i % 10)
            # 정렬 전 그래프: 프로 원본 인덱스 기준
            try:
                orig_idx = pro_df[pro_df['frame'] == frame_val].index[0]
                ax1.axvline(x=orig_idx, color=color, linestyle=':', linewidth=1.5)
                ax1.text(orig_idx, ax1.get_ylim()[1] * 0.95, name,
                         rotation=45, color=color, fontsize=8, fontweight='bold')
            except IndexError:
                pass
            # 정렬 후 그래프: DTW 경로 인덱스 기준
            if frame_val in frame_to_path_idx:
                pidx = frame_to_path_idx[frame_val]
                ax2.axvline(x=pidx, color=color, linestyle=':', linewidth=1.5)
                ax2.text(pidx, ax2.get_ylim()[1] * 0.95, name,
                         rotation=45, color=color, fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.show()


def make_simulated_user(pro_path: str, output_path: str, speed_ratio: float = 0.8, noise_std: float = 8.0, seed: int = 42):
    """
    실제 유저 데이터가 없을 때 테스트용으로 프로 데이터를 변형합니다.
    speed_ratio: 프레임을 이 비율만큼 랜덤 서브샘플 (0.8 = 프로보다 빠른 스윙)
    noise_std  : 각도 컬럼에 추가할 노이즈의 표준편차 (단위: 도)
    """
    rng = np.random.default_rng(seed)
    df = pd.read_csv(pro_path).copy()

    # 각도에 노이즈 추가
    for col in ANGLE_COLS:
        if col in df.columns:
            df[col] = df[col] + rng.normal(0, noise_std, len(df))

    # r_wrist_y에도 소량 노이즈
    df['r_wrist_y'] = df['r_wrist_y'] + rng.normal(0, 0.015, len(df))

    # 서브샘플로 "다른 속도의 스윙" 시뮬레이션
    n = max(10, int(len(df) * speed_ratio))
    indices = sorted(rng.choice(len(df), n, replace=False))
    df = df.iloc[indices].reset_index(drop=True)
    df['frame'] = range(1, len(df) + 1)

    df.to_csv(output_path, index=False)
    print(f"[시뮬레이션] 유저 데이터 생성 완료: {output_path}  ({len(df)} 프레임)")
    return output_path


if __name__ == "__main__":
    import os

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PRO_PATH = os.path.join(ROOT, 'data', 'processed', 'tigerwoods_angle_enhanced.csv')
    USER_PATH = os.path.join(ROOT, 'data', 'processed', 'simulated_user_angle.csv')
    OUTPUT_PATH = os.path.join(ROOT, 'data', 'processed', 'dtw_aligned.csv')

    # 실제 유저 데이터가 있으면 USER_PATH를 해당 경로로 교체하세요
    if not os.path.exists(USER_PATH):
        make_simulated_user(PRO_PATH, USER_PATH, speed_ratio=0.8, noise_std=8.0)

    # DTW 정렬 실행
    aligned_df = align_swings(PRO_PATH, USER_PATH, feature='r_wrist_y', window_ratio=0.2)
    aligned_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n정렬 결과 저장: {OUTPUT_PATH}")
    print(f"정렬된 행 수: {len(aligned_df)}")
    print("\n--- 앞부분 미리보기 ---")
    print(aligned_df[['pro_frame', 'user_frame', 'pro_r_elbow', 'user_r_elbow', 'diff_r_elbow']].head(10).to_string(index=False))

    # 시각화 (이벤트 마커 포함)
    events = {'Address': 53, 'Takeaway': 83, 'Top': 365, 'Impact': 489, 'Finish': 709}
    plot_alignment(PRO_PATH, USER_PATH, feature='r_wrist_y', events=events)
