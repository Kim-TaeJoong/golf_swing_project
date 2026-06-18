"""
check.py — 데이터 시각화 및 이벤트 검증 도구

역할:
  분석 중간에 데이터가 올바르게 처리되었는지 육안으로 확인하기 위한 시각화 스크립트.
  event_detector가 찾아낸 이정표 프레임을 그래프에 세로선으로 표시하여
  탐지 결과가 실제 동작과 일치하는지 검증한다.

주요 함수:
  - plt_csv              : 오른손목 y좌표의 위치·속도 그래프 출력 (초기 탐색용)
  - plot_selected_features: 지정 피처(각도·좌표)와 이벤트 구간을 함께 시각화
  - play_video_with_events: 원본 영상을 재생하며 현재 스윙 단계를 오버레이로 표시

사용법:
  파일 하단의 events 딕셔너리와 my_targets 리스트를 수정하여 원하는 피처를 확인한다.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2


# ── 1. 손목 위치·속도 그래프 (초기 탐색용) ────────────────────────────────────
def plt_csv(csv_path):
    """
    오른손목 y좌표의 원본·스무딩·속도(도함수) 그래프를 그린다.

    이벤트 탐지 알고리즘을 개발하기 전, 데이터의 전반적인 패턴과
    적절한 스무딩 윈도우 크기를 파악하기 위한 탐색적 분석 함수다.

    Args:
        csv_path : 분석할 각도·좌표 CSV 경로
    """
    df = pd.read_csv(csv_path)

    y = df['r_wrist_y']

    # 5프레임 중앙 이동평균으로 프레임 단위 노이즈를 제거한다.
    detect_df = y.rolling(window=5, center=True, min_periods=1).mean()

    # np.gradient: 중앙 차분법으로 속도를 계산. diff()보다 노이즈에 강하다.
    dy = np.gradient(detect_df)

    # 미분 과정에서 증폭된 노이즈를 추가 스무딩으로 제거한다.
    dy_smoothed = pd.Series(dy).rolling(window=7, center=True, min_periods=1).mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # 상단: 위치 그래프 (원본 vs 스무딩)
    ax1.plot(df['frame'], y, label='original position', alpha=0.4, color='gray')
    ax1.plot(df['frame'], detect_df, label='smoothed position (window=5)', linewidth=2, color='tab:blue')
    ax1.set_ylabel('r_wrist_y (Position)')
    ax1.set_title('Wrist Position and Velocity (Derivative)')
    ax1.legend()
    ax1.grid(True)

    # 하단: 속도 그래프 (0선 기준으로 방향 전환 지점을 시각적으로 확인)
    ax2.plot(df['frame'], dy_smoothed, label='velocity (dy/dt)', color='tab:orange', linewidth=2)
    ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('frame')
    ax2.set_ylabel('Velocity (Change in y)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


# ── 2. 지정 피처 + 이벤트 세로선 시각화 ─────────────────────────────────────
def plot_selected_features(csv_path, target_features, events=None):
    """
    지정한 피처(target_features)와 스윙 이벤트 구간을 함께 그래프로 표시한다.

    이정표 탐지 결과가 각 피처의 변화 패턴과 실제로 일치하는지 검증하거나,
    특정 각도·좌표값이 어떤 스윙 단계에서 어떻게 변하는지 분석할 때 사용한다.

    Args:
        csv_path        : 분석할 CSV 경로
        target_features : 그래프로 그릴 컬럼 이름 리스트 (예: ['r_wrist_y', 'x_factor'])
        events          : {이벤트명: 프레임번호} 딕셔너리. None이면 이벤트 선 없이 출력.
    """
    df = pd.read_csv(csv_path)

    for feature in target_features:
        # CSV에 없는 컬럼 이름을 입력한 경우 건너뜀
        if feature not in df.columns:
            print(f"경고: '{feature}' 컬럼이 데이터에 없습니다. 이름을 다시 확인하세요.")
            continue

        y = df[feature]

        # 데이터가 전부 NaN인 컬럼은 그릴 수 없으므로 건너뜀
        if y.isna().all():
            print(f"경고: '{feature}' 데이터가 모두 비어있습니다(NaN).")
            continue

        detect_df = y.rolling(window=5, center=True, min_periods=1).mean()

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(df['frame'], y, label=f'Original {feature}', alpha=0.4, color='gray')
        ax.plot(df['frame'], detect_df, label=f'Smoothed {feature}', linewidth=2, color='tab:blue')

        ax.set_xlabel('Frame')
        ax.set_ylabel(f'{feature} Value')
        ax.set_title(f'[{feature}] Position Analysis')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        # 이벤트 프레임마다 색상이 다른 세로 점선을 그려 이정표를 표시한다.
        if events:
            cmap = plt.get_cmap('tab10')
            for i, (event_name, frame_val) in enumerate(events.items()):
                color = cmap(i % 10)
                ax.axvline(x=frame_val, color=color, linestyle=':', linewidth=2)
                ax.text(frame_val, ax.get_ylim()[1] * 1.02, event_name,
                        rotation=45, color=color, fontweight='bold', ha='left')

        plt.tight_layout()

    print(f"현재 표시 중인 속성: {feature}")
    plt.show()


# ── 3. 영상 재생 + 이벤트 오버레이 ───────────────────────────────────────────
def play_video_with_events(video_path, events):
    """
    영상을 재생하면서 현재 스윙 단계(이벤트 이름)를 화면 상단에 오버레이로 표시한다.

    이정표 탐지 결과를 영상과 직접 대조하여 탐지 정확도를 눈으로 검증할 때 사용한다.
    이벤트 프레임에 도달하면 상태(current_status)를 업데이트하고,
    그 이후 프레임에도 계속 같은 상태를 표시한다(상태 유지 방식).

    Args:
        video_path : 재생할 영상 경로
        events     : {이벤트명: 프레임번호} 딕셔너리
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("영상을 열 수 없습니다.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    current_status = "Waiting for Start..."
    # {프레임번호: 이벤트명} 형태로 역변환하여 O(1) 조회 가능하게 한다.
    frame_to_event = {v: k for k, v in events.items()}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # 새로운 이벤트 프레임에 도달하면 현재 단계를 업데이트한다.
        if current_frame in frame_to_event:
            current_status = frame_to_event[current_frame]
            print(f"이벤트 감지: {current_status} (Frame {current_frame})")
            # 이벤트 순간에 잠깐 멈추고 싶다면 아래 주석을 해제한다.
            # cv2.waitKey(500)

        # 이벤트 텍스트를 매 프레임마다 화면에 그린다.
        # 검은 배경 박스를 먼저 그려서 배경색에 관계없이 텍스트가 잘 보이게 한다.
        cv2.rectangle(frame, (0, 0), (400, 80), (0, 0, 0), -1)
        cv2.putText(frame, f"STAGE: {current_status}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.putText(frame, f"F: {current_frame}", (20, height-20 if 'height' in locals() else 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('Golf Analysis', frame)

        # int(100/fps): 재생 속도를 영상 FPS에 맞게 조절
        # 슬로우모션으로 보려면 waitKey 값을 100 이상으로 늘린다.
        if cv2.waitKey(int(100/fps)) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ── 실행 ──────────────────────────────────────────────────────────────────────
# event_detector로 탐지한 최종 이정표 프레임 (검증 완료)
events = {
    'Address'  : 53,
    'Takeaway' : 83,
    'Top'      : 365,
    'Downswing': 366,
    'Impact'   : 489,
    'Finish'   : 709,
}

# 시각화할 피처 목록 (CSV 컬럼 이름과 일치해야 한다)
my_targets = ['r_wrist_y', 'x_factor']

plot_selected_features(
    'data/processed/tigerwoods_angle_enhanced.csv',
    target_features=my_targets,
    events=events
)

# 영상 재생 검증 (필요 시 주석 해제)
# play_video_with_events('data/processed/tiger_final_enhanced.mp4', events)
