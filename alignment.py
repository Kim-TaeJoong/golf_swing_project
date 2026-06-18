"""
alignment.py — 파이프라인 3단계: 스윙 이정표(milestone) 탐지 실행 진입점

역할:
  전처리된 각도·좌표 CSV 데이터를 읽어 event_detector에 넘기고,
  Address → Takeaway → Mid-Backswing → Top of Swing → Downswing
  → Impact → Follow-through → Finish 순서의 스윙 이정표 프레임을 추출한다.

  추출된 이정표는 DTW 정렬(dtw_aligner.py) 단계에서 프로-아마 구간을 맞추는 데 사용된다.

사용법:
  python alignment.py
  (data_path를 분석 대상 CSV 경로로 수정하면 다른 선수 데이터에도 적용 가능)
"""

import numpy as np
import pandas as pd
from event_detector import detect_swing_event
import os


def run_milestone_detection(file_path):
    """
    CSV 파일을 읽어 스윙 이정표를 탐지하고 결과를 출력한다.

    Args:
        file_path : 분석할 각도·좌표 CSV 경로
                    (preprocess → keypoint_extractor → analyzer를 거쳐 생성된 파일)

    Returns:
        milestones : {이벤트명: 프레임번호} 형태의 딕셔너리
                     탐지 실패 시 None 반환
    """
    print(f"--- [분석 시작] {file_path} ---")

    # 1. 데이터 로드
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        return None

    # 2. 이정표 탐색 (event_detector 호출)
    # 내부적으로 5프레임 스무딩을 거쳐 노이즈를 제거한 뒤 각 이벤트를 찾는다.
    milestones = detect_swing_event(df)

    # 3. 결과 출력 및 검증
    print("\n[추출된 스윙 이정표]")
    for event, frame in milestones.items():
        print(f"  {event:<15} : {frame:>4} 프레임")

    return milestones


if __name__ == "__main__":
    # 타이거 우즈 데이터를 기본 테스트 대상으로 사용
    data_path = 'data/processed/tigerwoods_angle_enhanced.csv'
    tiger_milestones = run_milestone_detection(data_path)
