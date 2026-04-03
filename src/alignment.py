import numpy as np
import pandas as pd
from event_detector import detect_swing_event
import os

def run_milestone_detection(file_path):
    print(f"--- [분석 시작] {file_path} ---")
    
    # 1. 데이터 로드
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        return None

    # 2. 이정표 탐색 (이벤트 디텍터 호출)
    # 내부적으로 5프레임 스무딩을 거쳐 안정적인 인덱스를 찾습니다.
    milestones = detect_swing_event(df)
    
    # 3. 결과 출력 및 검증
    print("\n[추출된 스윙 이정표]")
    for event, frame in milestones.items():
        print(f"📍 {event:<10} : {frame:>4} 프레임")
        
    return milestones


if __name__ == "__main__":
    # 타이거 우즈의 데이터 경로 (본인의 경로에 맞게 수정하세요)
    print("프로그램이 시작되었습니다!")
    data_path = 'data/processed/tigerwoods_angle_enhanced.csv'
    tiger_milestones = run_milestone_detection(data_path)