#시계열 데이터 기반 봇 탐지 데이터 셋 구축 및 시각화 프로그래밍
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.font_manager as fm
import pandas as pd

# 한글 폰트 설정
def setup_korean_font():
    """한글 폰트 설정 (수정 버전)"""
    font_path = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'

    # 1. 폰트 파일 존재 여부 확인 및 설치
    if not os.path.exists(font_path):
        print("Installing Nanum fonts...")
        os.system('sudo apt-get install -y fonts-nanum > /dev/null')

    # 2. 폰트 매니저에 폰트 추가 (문제의 원인 해결)
    # FontProperties 대신 font_manager.fontManager.addfont를 사용.
    fm.fontManager.addfont(font_path)

    # 3. 전역 설정 적용
    # 'NanumBarunGothic'은 나눔폰트 파일의 실제 이름입니다.
    plt.rc('font', family='NanumBarunGothic')
    plt.rcParams['axes.unicode_minus'] = False
    print("✅ 한글 폰트 설정 완료 (Matplotlib 연동 성공)")

# 샘플 데이터셋 구축
def build_bot_detection_dataset():
  # [1] 데이터 생성
  np.random.seed(42)

  # 시계열 데이터 봇 데이터
  bot_1 = np.random.normal(0,0.001, (50, 100))
  bot_2 = np.random.normal(0,0.001, (30, 100))
  all_bots = np.vstack((bot_1, bot_2)) # (80, 100)

  # 시계열 사람 데이터
  human = np.random.normal(0,0.05, (80, 100)) #(80, 100)

  # [2] 넘파이의 리덕션 집계
  # 모든 데이터셋을 합치기
  X_data = np.vstack((all_bots, human)) # (160, 100)
  # 합쳐진 2차원 배열의 통계 분포 : 평균, 표준편차
  row_means = np.mean(X_data, axis = 1).reshape(-1, 1) # (160, 1)
  row_stds = np.std(X_data, axis = 1).reshape(-1, 1) # (160, 1)

  # [3] 레이블 생성 (봇 = 1, 사람 = 0)
  y_data = np.concatenate([np.ones(80), np.zeros(80)]).reshape(-1,1) #(160,1)

  # [4] 최종 결합 : 데이터 (100) + 평균(1) + 표준편차(1) + 레이블(1) => (160, 103)
  dataset = np.column_stack(((X_data, row_means, row_stds, y_data)))

  # [5] 데이터 셔플링
  np.random.shuffle(dataset)
  print(dataset.shape) # (160, 103)
  print(dataset[:5, 100: ])
  print("✅ NumPy 리덕션 및 데이터셋 구축 완료:")

  return dataset

  #시각화
# 두 집단(봇 사람 표준편차 데이터)의 분포 비교 시각화
def visualize_by_numpy(dataset):
 # 1. 표준편차 샘플 추출 (160,)
 std_vals = dataset[:, -2]
 label_vals = dataset[:, -1] # label 샘플

 # 2. 논리 인덱싱을 이용하여 그룹 봇과 사람으로 분리
 bots_stds = std_vals[label_vals == 1]
 humans_stds = std_vals[label_vals == 0]
 
 # 3 시각화
 plt.figure(figsize=(10,6))
 plt.hist(bots_stds, bins=20, alpha=0.7, label='봇', color='red') # 봇 집단 표준편차 히스토그램
 plt.hist(humans_stds, bins=30, alpha=0.7, label='사람', color='blue', edgecolor='black')
 plt.title("표준편차에 따른 봇과 사람 분포 비교")
 plt.xlabel("표준편차")
 plt.ylabel("빈도수")
 plt.grid(True)
 plt.legend()
 plt.show()
