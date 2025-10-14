import polars as pl
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import gc
import os

# --- 1. 설정 (Configuration) ---
# 데이터 경로와 샘플링 비율을 설정합니다.
DATA_PATH = '/home/jjh/Project/competition/13_toss/data/'
TRAIN_FILE = 'train.parquet'
SAMPLE_FRACTION = 0.02 # GPU 메모리에 맞게 전체 데이터의 20%만 사용 (조정 가능)
N_TRIALS = 50 # Optuna 탐색 횟수

def feature_engineering(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    전략 계획서에 기반한 피처 엔지니어링을 수행합니다.
    """
    print("피처 엔지니어링을 시작합니다...")

    # --- NEW: Robust Type Casting at the beginning ---
    # 'hour' 및 'history_*' 컬럼들이 문자열일 수 있으므로, 숫자 타입으로 명시적 변환
    schema = df.collect_schema()
    numeric_cols_to_cast = ['hour']
    numeric_cols_to_cast += [col for col in schema.names() if 'history_' in col]
    
    df = df.with_columns(
        [pl.col(c).cast(pl.Float64, strict=False) for c in numeric_cols_to_cast]
    )

    # A. 전처리: 수치형 피처 로그 변환
    history_a_feats = [col for col in schema.names() if 'history_a_' in col]
    # 여러 컬럼에 대한 연산을 리스트로 묶어 한 번에 처리
    df = df.with_columns(
        [pl.col(feat).log1p() for feat in history_a_feats]
    )

    # B. 시간/순서 기반 피처 (이제 'hour'는 숫자 타입이므로 안전)
    df = df.with_columns([
        pl.col('day_of_week').is_in(['6', '7']).cast(pl.Int8).alias('is_weekend'),
        (np.pi * pl.col('hour') / 12).sin().alias('hour_sin'),
        (np.pi * pl.col('hour') / 12).cos().alias('hour_cos'),
    ])

    # C. 상호작용 피처
    df = df.with_columns([
        (pl.col('age_group').cast(str) + "_" + pl.col('gender').cast(str)).alias('age_x_gender')
    ])
    
    # --- seq 피처 분석 기반 모델링 강화 전략 적용 ---
    print("seq 피처 분석 기반 전략을 적용합니다...")
    
    # [A. 세션 레벨 피처]
    df = df.with_columns(
        pl.col('seq').str.split(',').list.len().alias('seq_length')
    )
    
    # 나머지 세션 레벨 피처 추가
    df = df.with_columns([
        pl.col('seq_length').log1p().alias('log_seq_length'),
        (pl.col('seq_length') < 50).cast(pl.Int8).alias('is_very_short_session'),
        pl.when(pl.col('seq_length') <= 50).then(pl.lit('0-50'))
          .when(pl.col('seq_length') <= 200).then(pl.lit('51-200'))
          .when(pl.col('seq_length') <= 500).then(pl.lit('201-500'))
          .otherwise(pl.lit('500+'))
          .alias('seq_length_binned')
    ])

    # [D. 사용자 기반 피처 (기존 + seq 강화)]
    user_agg = df.group_by('feat_c_2').agg([
        pl.len().alias('user_total_exposure'),
        pl.col('clicked').sum().alias('user_total_clicks'),
        pl.col('inventory_id').n_unique().alias('user_unique_inventories'),
        pl.col('seq_length').mean().alias('user_mean_session_length'),
        pl.col('seq_length').max().alias('user_max_session_length'),
        pl.col('seq_length').std().alias('user_session_length_std'),
    ])
    
    user_agg = user_agg.with_columns(
        (pl.col('user_total_clicks') / pl.col('user_total_exposure')).alias('user_ctr')
    )

    df = df.join(user_agg, on='feat_c_2', how='left')

    print("피처 엔지니어링 완료.")
    return df

def objective(trial: optuna.Trial, X: pl.DataFrame, y: pl.Series, categorical_features: list, scale_pos_weight: float) -> float:
    """
    Optuna가 최적화할 목적 함수입니다.
    """
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_estimators': 1000,
        # --- 수정된 부분: GPU 설정을 주석 처리하여 CPU를 사용하도록 변경 ---
        # 'device': 'gpu', 
        # 'gpu_platform_id': 0,
        # 'gpu_device_id': 0,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 200),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'scale_pos_weight': scale_pos_weight
    }

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = lgb.LGBMClassifier(**params)
    
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='auc',
              callbacks=[lgb.early_stopping(100, verbose=False)],
              categorical_feature=categorical_features)

    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)
    return auc

def main():
    """
    메인 실행 함수
    """
    print(f"데이터 로드를 시작합니다. 전체 데이터의 {SAMPLE_FRACTION * 100}%를 샘플링합니다...")
    lazy_df = pl.scan_parquet(os.path.join(DATA_PATH, TRAIN_FILE))

    random_index_col = 'random_index'
    sampled_df = lazy_df.with_columns(
        (pl.int_range(0, pl.len()).shuffle().over('clicked')).alias(random_index_col)
    ).filter(
        pl.col(random_index_col) < pl.len() * SAMPLE_FRACTION
    ).drop(random_index_col).collect()

    print(f"데이터 샘플링 완료. 총 {len(sampled_df)}개 행")
    
    processed_df = feature_engineering(sampled_df.lazy()).collect()

    TARGET = 'clicked'
    categorical_features = []
    for col, dtype in processed_df.schema.items():
        if (dtype == pl.String) or (dtype == pl.Categorical) or (str(dtype) == 'Enum'):
            if str(dtype) != 'Categorical':
                 processed_df = processed_df.with_columns(pl.col(col).cast(pl.Categorical))
            categorical_features.append(col)

    y = processed_df.get_column(TARGET)
    X = processed_df.drop(TARGET)

    value_counts = y.value_counts()
    count_0 = value_counts.filter(pl.col(TARGET) == 0)['count'][0]
    count_1 = value_counts.filter(pl.col(TARGET) == 1)['count'][0]
    scale_pos_weight = count_0 / count_1
    print(f"scale_pos_weight 계산 완료: {scale_pos_weight:.2f}")
    
    X_pd = X.to_pandas()
    y_pd = y.to_pandas()
    
    del sampled_df, processed_df, X, y
    gc.collect()

    print(f"\nOptuna를 이용한 하이퍼파라미터 최적화를 시작합니다. (n_trials={N_TRIALS})")
    study = optuna.create_study(direction='maximize', study_name='lgbm_ctr_tuning')
    
    func = lambda trial: objective(trial, X_pd, y_pd, categorical_features, scale_pos_weight)
    
    study.optimize(func, n_trials=N_TRIALS)

    print("\n최적화 완료!")
    print(f"최고 점수 (AUC): {study.best_value:.5f}")
    print("최적 하이퍼파라미터:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

if __name__ == '__main__':
    main()

