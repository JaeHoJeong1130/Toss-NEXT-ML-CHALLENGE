# 파일명: preprocess.py
import polars as pl
import gc

dPATH = '/home/jjh/Project/competition/13_toss/data/'

def feature_engineering(df: pl.LazyFrame) -> pl.LazyFrame:
    """모든 전처리가 완료되도록 수정한 최종 피처 엔지니어링 함수"""
    
    # 1. 타입 변환
    string_to_int_cols = ['gender', 'age_group', 'inventory_id', 'day_of_week', 'hour']
    df = df.with_columns([
        pl.col(c).cast(pl.Float64).cast(pl.Int64) for c in string_to_int_cols
    ])
    
    # 2. 'seq' 피처 생성
    df = df.with_columns(
        pl.col('seq').str.split(by=',')
    ).with_columns(
        pl.col('seq').list.len().alias('seq_length'),
        pl.col('seq').list.n_unique().alias('seq_unique_count')
    ).drop('seq')

    # 3. 상호작용 피처
    df = df.with_columns([
        (pl.col("gender").cast(str) + "_" + pl.col("age_group").cast(str)).alias("gender_age_interaction"),
        (pl.col("age_group").cast(str) + "_" + pl.col("inventory_id").cast(str)).alias("age_inventory_interaction"),
    ])
    
    # 4. 통계 피처
    user_agg = df.group_by("l_feat_1").agg([
        pl.len().alias("user_exposure_count"),
        pl.n_unique("inventory_id").alias("user_unique_inventory_count"),
    ])
    df = df.join(user_agg, on="l_feat_1", how="left")
    
    # 5. 결측치 처리
    df = df.with_columns([
        pl.col('feat_e_3').fill_null(-1.0),
        pl.col(['gender', 'age_group']).fill_null(-1),
    ])
    df = df.fill_null(0.0)
    
    # ✨ --- 핵심 수정 사항 시작 --- ✨
    # DeepFM에 사용될 범주형 피처들의 인덱스를 0 이상으로 조정합니다.
    # (-1은 0으로, 0은 1로, ... 모든 값이 1씩 증가)
    deepfm_cat_features = ['gender', 'age_group', 'inventory_id', 'day_of_week', 'hour']
    df = df.with_columns([
        pl.col(c) + 1 for c in deepfm_cat_features
    ])
    # ✨ --- 핵심 수정 사항 끝 --- ✨

    # 6. 로그 변환
    for col in ['history_a_1', 'seq_length', 'seq_unique_count']:
        df = df.with_columns(
            (pl.col(col) + 1).log().alias(f"log_{col}")
        )
    df = df.with_columns(
        (pl.col('history_a_2') - pl.col('history_a_2').min() + 1).log().alias('log_history_a_2')
    )
        
    return df

def main():
    print("--- 1단계: 데이터 전처리 및 저장 시작 ---")
    
    print("학습 데이터 처리 중...")
    train_lazy = pl.scan_parquet(dPATH + 'train.parquet')
    train_processed_lazy = feature_engineering(train_lazy)
    categorical_cols_new = ['gender_age_interaction', 'age_inventory_interaction']
    for col in categorical_cols_new:
        train_processed_lazy = train_processed_lazy.with_columns(pl.col(col).cast(pl.Categorical))
    train_processed_lazy.sink_parquet(dPATH + 'train_processed.parquet')
    print(f"전처리된 학습 데이터가 '{dPATH}train_processed.parquet'에 저장되었습니다.")
    del train_lazy, train_processed_lazy
    gc.collect()

    print("\n테스트 데이터 처리 중...")
    test_lazy = pl.scan_parquet(dPATH + 'test.parquet')
    test_processed_lazy = feature_engineering(test_lazy)
    for col in categorical_cols_new:
        test_processed_lazy = test_processed_lazy.with_columns(pl.col(col).cast(pl.Categorical))
    test_processed_lazy.sink_parquet(dPATH + 'test_processed.parquet')
    print(f"전처리된 테스트 데이터가 '{dPATH}test_processed.parquet'에 저장되었습니다.")
    
    print("\n--- 모든 데이터 전처리 완료! ---")

if __name__ == '__main__':
    main()