# -*- coding: utf-8 -*-
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import QuantileTransformer

warnings.filterwarnings('ignore')

# --- Matplotlib 한글 폰트 설정 ---
import matplotlib.font_manager as fm

# 시스템에 설치된 폰트 확인
def get_korean_font():
    """한글 폰트 찾기"""
    font_list = [f.name for f in fm.fontManager.ttflist]
    
    # 우선순위대로 폰트 검색
    korean_fonts = ['NanumGothic', 'NanumBarunGothic', 'NanumSquare', 
                    'Noto Sans CJK KR', 'Malgun Gothic', 'AppleGothic']
    
    for font in korean_fonts:
        if font in font_list:
            print(f"✓ 한글 폰트 '{font}' 찾음!")
            return font
    
    print("⚠ 한글 폰트를 찾을 수 없습니다.")
    return None

# 폰트 설정
korean_font = get_korean_font()

if korean_font:
    # matplotlib 기본 설정
    plt.rcParams.update({
        'font.family': korean_font,
        'axes.unicode_minus': False,
        'font.size': 10,
    })
else:
    print("한글 폰트 설치 필요: sudo apt-get install fonts-nanum")

# seaborn 스타일 적용 (폰트 설정 후에!)
plt.style.use('seaborn-v0_8-whitegrid')

# 스타일 적용 후 폰트 재설정 (중요!)
if korean_font:
    plt.rcParams['font.family'] = korean_font
    plt.rcParams['axes.unicode_minus'] = False

print(f"현재 설정된 폰트: {plt.rcParams['font.family']}")


# --- 데이터 경로 설정 ---
# 데이터 파일이 있는 경로를 지정해주세요.
RAW_TRAIN_PATH = "/home/jjh/Project/competition/13_toss/data/train.parquet"
ENGINEERED_TRAIN_PATH = "/home/jjh/Project/competition/13_toss/data/new_data/new_train_2.parquet"
OUTPUT_DIR = "/home/jjh/Project/competition/13_toss/report_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# RAW_TRAIN_PATH = "./Toss/train.parquet"
# ENGINEERED_TRAIN_PATH = "./Toss/new_data/new_train_2.parquet"
# OUTPUT_DIR = "./report_figures"
# os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- 1. 데이터 분석 및 물리적 제약 식별용 그래프 ---

def plot_section_1_figures(raw_df):
    """1번 섹션: 원본 데이터 분석용 그래프들을 생성합니다."""
    
    # 폰트 재설정 (seaborn이 폰트를 초기화할 수 있음)
    plt.rcParams['font.family'] = get_korean_font() or 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    
    print("1번 섹션 그래프 생성 중...")

    # 1.1: 타겟 변수 분포 (클래스 불균형 확인)
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x='clicked', data=raw_df, palette='pastel')
    plt.title('타겟 변수(clicked)의 분포', fontsize=16, pad=15)
    plt.xlabel('클릭 여부 (0: 클릭 안함, 1: 클릭)', fontsize=12)
    plt.ylabel('데이터 수', fontsize=12)
    for p in ax.patches:
        ax.annotate(f'{p.get_height():,}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=11, color='black', xytext=(0, 10),
                    textcoords='offset points')
    plt.savefig(os.path.join(OUTPUT_DIR, '1-1_target_distribution.png'), dpi=300)
    print("  - 1-1_target_distribution.png 저장 완료")

    # 1.2: `seq` 컬럼 길이 분포 (비정형성의 시각화)
    seq_lengths = raw_df['seq'].str.split(',').str.len().fillna(0)
    plt.figure(figsize=(10, 6))
    sns.histplot(seq_lengths, bins=50, kde=True)
    plt.title('`seq` 컬럼의 길이 분포', fontsize=16, pad=15)
    plt.xlabel('시퀀스 길이', fontsize=12)
    plt.ylabel('빈도', fontsize=12)
    plt.xlim(0, seq_lengths.quantile(0.99)) # 상위 1% 이상은 제외하여 가독성 확보
    plt.savefig(os.path.join(OUTPUT_DIR, '1-2_seq_length_distribution.png'), dpi=300)
    print("  - 1-2_seq_length_distribution.png 저장 완료")

    # 1.3: 시간 관련 피처 분포 (주기성 확인)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.countplot(x='hour', data=raw_df, ax=axes[0], palette='viridis')
    axes[0].set_title('시간(hour)별 데이터 분포', fontsize=14)
    axes[0].set_xlabel('시간 (0-23시)', fontsize=12)
    axes[0].set_ylabel('데이터 수', fontsize=12)

    sns.countplot(x='day_of_week', data=raw_df, ax=axes[1], palette='plasma')
    axes[1].set_title('요일(day_of_week)별 데이터 분포', fontsize=14)
    axes[1].set_xlabel('요일 (0-6)', fontsize=12)
    axes[1].set_ylabel('')
    fig.suptitle('시간 관련 피처의 분포', fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(OUTPUT_DIR, '1-3_time_feature_distribution.png'), dpi=300)
    print("  - 1-3_time_feature_distribution.png 저장 완료")
    plt.close('all')

# --- 2. 데이터 정제 및 피처 엔지니어링용 그래프 ---

def plot_section_2_figures(raw_df, engineered_df):
    """2번 섹션: 피처 엔지니어링 효과 시각화용 그래프들을 생성합니다."""
    
    # 폰트 재설정
    plt.rcParams['font.family'] = get_korean_font() or 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    
    print("\n2번 섹션 그래프 생성 중...")

    # 2.1: 주기성 피처 변환 결과 (Sin/Cos 변환)
    hour_df = raw_df[['hour']].drop_duplicates().sort_values('hour').reset_index(drop=True)
    # 🔧 hour 컬럼을 숫자형으로 변환
    hour_df['hour'] = pd.to_numeric(hour_df['hour'], errors='coerce')
    hour_df = hour_df.dropna()  # 변환 실패한 행 제거
    hour_df['sin_hour'] = np.sin(2 * np.pi * hour_df['hour'] / 24)
    hour_df['cos_hour'] = np.cos(2 * np.pi * hour_df['hour'] / 24)

    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=hour_df, x='sin_hour', y='cos_hour', hue='hour', palette='twilight_shifted', s=150)
    plt.title('시간(hour) 피처의 Sin/Cos 주기성 변환', fontsize=16, pad=15)
    plt.xlabel('Sin(Hour)', fontsize=12)
    plt.ylabel('Cos(Hour)', fontsize=12)
    plt.legend(title='Hour', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.axis('equal')
    plt.savefig(os.path.join(OUTPUT_DIR, '2-1_cyclical_feature_transform.png'), dpi=300, bbox_inches='tight')
    print("  - 2-1_cyclical_feature_transform.png 저장 완료")

    # 2.2: 수치형 피처 정규화 효과 (QuantileTransformer)
    # 원본 데이터에서 왜도가 높은 피처를 하나 선택 (예: history_a_1)
    feature_to_normalize = 'history_a_1'
    if feature_to_normalize in raw_df.columns and feature_to_normalize in engineered_df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        sns.kdeplot(raw_df[feature_to_normalize].dropna(), ax=axes[0], fill=True, color='skyblue')
        axes[0].set_title('정규화 이전 분포 (원본)', fontsize=14)
        axes[0].set_xlabel(f'원본 {feature_to_normalize} 값', fontsize=12)
        axes[0].set_ylabel('밀도', fontsize=12)

        sns.kdeplot(engineered_df[feature_to_normalize].dropna(), ax=axes[1], fill=True, color='salmon')
        axes[1].set_title('정규화 이후 분포 (Quantile Transformed)', fontsize=14)
        axes[1].set_xlabel(f'변환된 {feature_to_normalize} 값', fontsize=12)
        axes[1].set_ylabel('')
        fig.suptitle(f'"{feature_to_normalize}" 피처의 정규화 효과 비교', fontsize=18)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(OUTPUT_DIR, '2-2_normalization_effect.png'), dpi=300)
        print("  - 2-2_normalization_effect.png 저장 완료")

    # 2.3: 타겟 인코딩 효과 (Target Encoding)
    # 예시로 'gender' 피처를 사용
    if 'gender_te' in engineered_df.columns:
        plt.figure(figsize=(10, 7))
        sns.boxenplot(x='clicked', y='gender_te', data=engineered_df, palette='coolwarm')
        plt.title("'gender' 피처의 타겟 인코딩(Target Encoding) 효과", fontsize=16, pad=15)
        plt.xlabel('클릭 여부', fontsize=12)
        plt.ylabel('타겟 인코딩된 값 (gender_te)', fontsize=12)
        plt.savefig(os.path.join(OUTPUT_DIR, '2-3_target_encoding_effect.png'), dpi=300)
        print("  - 2-3_target_encoding_effect.png 저장 완료")
    
    plt.close('all')

# --- 메인 실행 함수 ---
def main():
    """데이터를 로드하고 모든 시각화 자료를 생성합니다."""
    
    print("데이터 로딩 시작...")
    try:
        # 메모리 절약을 위해 일부 컬럼만 로드하고 샘플링
        raw_cols = ['clicked', 'seq', 'hour', 'day_of_week', 'history_a_1', 'gender']
        raw_df = pd.read_parquet(RAW_TRAIN_PATH, columns=raw_cols)
        if len(raw_df) > 500000:
             raw_df = raw_df.sample(n=500000, random_state=42).reset_index(drop=True)

        engineered_cols = ['clicked', 'history_a_1', 'gender_te']
        engineered_df = pd.read_parquet(ENGINEERED_TRAIN_PATH, columns=engineered_cols)
        if len(engineered_df) > 500000:
            engineered_df = engineered_df.sample(n=500000, random_state=42).reset_index(drop=True)

        print("데이터 로딩 완료.")
    except FileNotFoundError as e:
        print(f"오류: 데이터 파일을 찾을 수 없습니다. 경로를 확인해주세요. ({e})")
        return

    # 섹션 1 그래프 생성
    plot_section_1_figures(raw_df)

    # 섹션 2 그래프 생성
    plot_section_2_figures(raw_df, engineered_df)
    
    print(f"\n모든 그래프가 '{OUTPUT_DIR}' 폴더에 저장되었습니다.")


if __name__ == '__main__':
    main()

