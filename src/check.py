import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2

def plt_csv(csv_path):
    #csv_path = os.path.join('data', 'processed', 'tigerwoods_angle_enhanced.csv')
    df = pd.read_csv(csv_path)

    # 1. 위치 데이터 및 스무딩 (기존 코드 유지)
    y = df['r_wrist_y']
    detect_df = y.rolling(window=5, center=True, min_periods=1).mean()

    # 2. 도함수(속도) 계산
    # np.gradient는 중앙 차분법을 사용하여 diff()보다 노이즈에 조금 더 강합니다.
    dy = np.gradient(detect_df) 

    # 3. 속도 데이터 추가 스무딩 (미분 시 발생하는 노이즈 제거)
    dy_smoothed = pd.Series(dy).rolling(window=7, center=True, min_periods=1).mean()

    # --- 그래프 그리기 ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # 첫 번째 그래프: 위치 (Original vs Smoothed)
    ax1.plot(df['frame'], y, label='original position', alpha=0.4, color='gray')
    ax1.plot(df['frame'], detect_df, label='smoothed position (window=5)', linewidth=2, color='tab:blue')
    ax1.set_ylabel('r_wrist_y (Position)')
    ax1.set_title('Wrist Position and Velocity (Derivative)')
    ax1.legend()
    ax1.grid(True)

    # 두 번째 그래프: 도함수 (Velocity)
    ax2.plot(df['frame'], dy_smoothed, label='velocity (dy/dt)', color='tab:orange', linewidth=2)
    ax2.axhline(0, color='black', linestyle='--', alpha=0.5) # 속도 0 지점
    ax2.set_xlabel('frame')
    ax2.set_ylabel('Velocity (Change in y)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def plot_selected_features(csv_path, target_features, events=None):
    """
    사용자가 지정한(target_features) 속성만 골라서 그래프로 띄웁니다.
    """
    df = pd.read_csv(csv_path)
    
    # 지정한 속성들만 반복해서 그리기
    for feature in target_features:
        # 1. 오타 방지 (CSV에 없는 이름을 적었을 경우)
        if feature not in df.columns:
            print(f"⚠️ 경고: '{feature}' 컬럼이 데이터에 없습니다. 이름을 다시 확인하세요.")
            continue
            
        y = df[feature]
        
        # 데이터가 비어있으면 건너뛰기
        if y.isna().all():
            print(f"⚠️ 경고: '{feature}' 데이터가 모두 비어있습니다(NaN).")
            continue

        # --- 데이터 처리 ---
        detect_df = y.rolling(window=5, center=True, min_periods=1).mean()

        # --- 그래프 그리기 ---
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(df['frame'], y, label=f'Original {feature}', alpha=0.4, color='gray')
        ax.plot(df['frame'], detect_df, label=f'Smoothed {feature}', linewidth=2, color='tab:blue')
        
        ax.set_xlabel('Frame')
        ax.set_ylabel(f'{feature} Value')
        ax.set_title(f'[{feature}] Position Analysis')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        # --- 특정 프레임(이벤트)에 세로선 긋기 ---
        if events:
            cmap = plt.get_cmap('tab10') 
            for i, (event_name, frame_val) in enumerate(events.items()):
                color = cmap(i % 10)
                
                ax.axvline(x=frame_val, color=color, linestyle=':', linewidth=2)
                ax.text(frame_val, ax.get_ylim()[1] * 1.02, event_name, 
                         rotation=45, color=color, fontweight='bold', ha='left')

        plt.tight_layout()
        
        # --- 화면에 출력 ---
         # 창을 닫으면 다음 지정 속성으로 넘어갑니다.
    print(f"📊 현재 표시 중인 속성: {feature}")
    plt.show()

    
def play_video_with_events(video_path, events):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ 영상을 열 수 없습니다.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # ⭐ 현재 어떤 상태인지 기억해둘 변수 (상태 유지의 핵심)
    current_status = "Waiting for Start..."
    frame_to_event = {v: k for k, v in events.items()}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # 1. 새로운 이벤트 프레임에 도달하면 상태를 업데이트합니다.
        if current_frame in frame_to_event:
            current_status = frame_to_event[current_frame]
            print(f"🎯 이벤트 감지: {current_status} (Frame {current_frame})")
            
            # ⭐ 팁: 이벤트 순간에 영상을 아주 잠깐 멈추고 싶다면 아래 주석을 푸세요.
            # cv2.waitKey(500) # 0.5초 멈춤

        # 2. 업데이트된 current_status를 '매 프레임'마다 그립니다.
        # 화면 상단에 검은색 배경 박스를 넣으면 글자가 더 잘 보입니다.
        cv2.rectangle(frame, (0, 0), (400, 80), (0, 0, 0), -1) 
        cv2.putText(frame, f"STAGE: {current_status}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        cv2.putText(frame, f"F: {current_frame}", (20, height-20 if 'height' in locals() else 450), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('Golf Analysis', frame)

        # 재생 속도가 너무 빠르면 1000/fps 뒤에 숫자를 더해서 느리게 보세요.
        # 예: cv2.waitKey(100) -> 아주 느린 슬로우 모션
        if cv2.waitKey(int(100/fps)) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 실행 부분

'''
Address    :   53 프레임
📍 Takeaway   :   83 프레임
📍 Top of Swing :  527 프레임
📍 Downswing  :  611 프레임
📍 Impact     :  683 프레임
📍 Finish     :  709 프레임

 Address    :   53 프레임
📍 Takeaway   :   83 프레임
📍 Top of Swing :  178 프레임
📍 Downswing  :  178 프레임
📍 Impact     :  474 프레임
📍 Finish     :  709 프레임

 Address    :   53 프레임
📍 Takeaway   :   83 프레임
📍 Top of Swing :  365 프레임
📍 Downswing  :  366 프레임
📍 Impact     :  474 프레임
📍 Finish     :  709 프레임

 Address    :   54 프레임
📍 Takeaway   :   84 프레임
📍 Top of Swing :  365 프레임
📍 Downswing  :  366 프레임
📍 Impact     :  489 프레임
📍 Finish     :  709 프레임
'''
#plt_csv('data/processed/tigerwoods_angle_enhanced.csv')
events = {'Address': 53, 'Takeaway' : 83 ,'Top': 365, 'Downswing' : 366,'Impact': 489, 'Finish' : 709}
my_targets = ['r_wrist_y', 'x_factor']
plot_selected_features('data/processed/tigerwoods_angle_enhanced.csv',target_features=my_targets, events=events)
#play_video_with_events('data/processed/tiger_final_enhanced.mp4', events)