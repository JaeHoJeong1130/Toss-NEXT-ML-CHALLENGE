import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc

# --- 설정 (Configuration) ---
DATA_PATH = '/home/jjh/Project/competition/13_toss/data/'
TRAIN_FILE = 'train.parquet'
OUTPUT_DIR = 'seq_analysis_plots_v2' # 새로운 분석이므로 폴더 이름 변경
# 전체 데이터를 분석하기에는 너무 크므로, 분석을 위해 10% 샘플 사용
SAMPLE_FRACTION = 0.1 

def main():
    """
    'seq' (유저 서버 로그 시퀀스) 피처를 심층 분석하는 스크립트.
    'seq'가 콤마로 구분된 문자열임을 반영하여 분석 전략을 수정.
    """
    # 0. 준비 단계
    print("스크립트를 시작합니다.")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    print(f"'{OUTPUT_DIR}' 폴더에 분석 결과를 저장합니다.")

    # 데이터 로드 (분석을 위해 샘플링)
    print(f"데이터 로드를 시작합니다. 전체 데이터의 {SAMPLE_FRACTION * 100}%를 샘플링합니다...")
    try:
        lazy_df = pl.scan_parquet(os.path.join(DATA_PATH, TRAIN_FILE))
        
        # --- 수정된 부분: 'seq'에서 'seq_length' 피처 추출 ---
        # 'seq'는 콤마로 구분된 문자열이므로, 길이를 세어 'seq_length'라는 새로운 피처를 생성합니다.
        lazy_df = lazy_df.with_columns(
            pl.col('seq').str.split(',').list.len().alias('seq_length')
        )

        df = lazy_df.filter(
            pl.int_range(0, pl.len()).shuffle(seed=42) < (pl.len() * SAMPLE_FRACTION)
        ).collect()

        print(f"데이터 샘플링 완료. 총 {len(df)}개 행")
    except FileNotFoundError:
        print(f"오류: '{TRAIN_FILE}' 파일을 찾을 수 없습니다.")
        return

    # 1. 'seq_length' 값의 전체 분포 분석
    print("\n[1/3] 'seq_length' 피처의 전체 분포를 분석 중...")
    seq_len_description = df.select('seq_length').describe().to_pandas()
    print("`seq_length` 피처 기본 통계량:")
    print(seq_len_description)

    plt.figure(figsize=(12, 6))
    upper_quantile = df['seq_length'].quantile(0.99)
    sns.histplot(df.filter(pl.col('seq_length') <= upper_quantile)['seq_length'], bins=50, kde=True)
    plt.title(f'Distribution of Session Length (seq_length) (up to 99th percentile: {upper_quantile})', fontsize=15)
    plt.xlabel('Session Length', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.savefig(os.path.join(OUTPUT_DIR, '1_seq_length_distribution.png'))
    plt.close()

    # 2. 'seq_length'에 따른 클릭률(CTR) 변화 분석
    print("\n[2/3] 'seq_length'에 따른 CTR 변화를 분석 중...")
    # seq_length를 10 단위로 그룹화(binning)하여 분석
    seq_len_bins_df = df.with_columns(
        (pl.col('seq_length') // 10 * 10).alias('seq_length_bin')
    ).group_by('seq_length_bin').agg([
        pl.col('clicked').mean().alias('ctr'),
        pl.len().alias('count')
    ]).sort('seq_length_bin')

    # 데이터 수가 충분한 구간만 시각화
    min_count_threshold = seq_len_bins_df['count'].quantile(0.01)
    seq_len_bins_pd = seq_len_bins_df.filter(
        (pl.col('count') > min_count_threshold) & (pl.col('seq_length_bin') <= upper_quantile)
    ).to_pandas()
    
    plt.figure(figsize=(16, 7))
    sns.lineplot(x='seq_length_bin', y='ctr', data=seq_len_bins_pd, marker='o')
    plt.title('CTR by Session Length Bins', fontsize=15)
    plt.xlabel('Session Length Bin (e.g., 0 means length 0-9)', fontsize=12)
    plt.ylabel('Click-Through Rate', fontsize=12)
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, '2_ctr_by_seq_length_bins.png'))
    plt.close()

    # 3. 사용자별 최대 'seq_length' 값 분포 분석
    print("\n[3/3] 사용자별 최대 'seq_length' 값 분포를 분석 중...")
    user_id = 'feat_c_2' # 사용자 ID로 추정되는 피처
    user_max_seq_len = df.group_by(user_id).agg(
        pl.col('seq_length').max().alias('user_max_seq_length')
    )
    
    user_max_seq_len_pd = user_max_seq_len.to_pandas()

    plt.figure(figsize=(12, 6))
    upper_quantile_user = user_max_seq_len_pd['user_max_seq_length'].quantile(0.99)
    sns.histplot(user_max_seq_len_pd[user_max_seq_len_pd['user_max_seq_length'] <= upper_quantile_user]['user_max_seq_length'], bins=50, kde=True)
    plt.title(f'Distribution of User Max Session Length (up to 99th percentile: {upper_quantile_user})', fontsize=15)
    plt.xlabel('User Max Session Length', fontsize=12)
    plt.ylabel('Number of Users', fontsize=12)
    plt.savefig(os.path.join(OUTPUT_DIR, '3_user_max_seq_length_distribution.png'))
    plt.close()

    print("\n모든 분석이 완료되었습니다.")
    print(f"결과는 '{OUTPUT_DIR}' 폴더에서 확인하실 수 있습니다.")


if __name__ == '__main__':
    main()

