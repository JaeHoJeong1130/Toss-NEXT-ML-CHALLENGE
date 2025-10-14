import polars as pl
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import gc
import os
import pyarrow.parquet as pq # 테스트 데이터 처리를 위해 pyarrow 추가

# --- 1. 설정 (Configuration) ---
DATA_PATH = '/home/jjh/Project/competition/13_toss/data/'
TRAIN_FILE = 'train.parquet'
TEST_FILE = 'test.parquet'
SUBMISSION_FILE = 'submission.csv'

# --- ★★★ 모드 설정 ★★★ ---
# True: Optuna로 하이퍼파라미터 탐색 수행
# False: 저장된 최적 파라미터로 최종 모델 학습 및 제출 파일 생성
RUN_OPTUNA = False 

# Optuna를 실행하지 않을 경우, 아래에 찾은 최적 파라미터를 입력하세요.
BEST_PARAMS = {
    'learning_rate': 0.01953050646886217,
    'num_leaves': 197,
    'max_depth': 11,
    'min_child_samples': 47,
    'subsample': 0.9784859995445454,
    'colsample_bytree': 0.687904324587539,
    'reg_alpha': 1.1884479611933629e-07,
    'reg_lambda': 1.7791398376710305e-05
}

# 최종 모델 학습 및 Optuna 탐색에 사용할 데이터 샘플링 비율
# 최종 모델 학습 시 메모리가 부족하면 이 값을 줄여주세요.
SAMPLE_FRACTION = 0.1 
N_TRIALS = 50 # Optuna 탐색 횟수

def feature_engineering(df: pl.LazyFrame, user_agg: pl.DataFrame = None, is_train: bool = True) -> pl.LazyFrame:
    """
    전략 계획서에 기반한 피처 엔지니어링을 수행합니다.
    """
    if is_train:
        print("학습 데이터 피처 엔지니어링을 시작합니다...")
    else:
        # 테스트 모드에서는 user_agg가 필수로 제공되어야 합니다.
        pass

    schema = df.collect_schema()
    numeric_cols_to_cast = ['hour']
    numeric_cols_to_cast += [col for col in schema.names() if 'history_' in col]
    
    df = df.with_columns(
        [pl.col(c).cast(pl.Float64, strict=False) for c in numeric_cols_to_cast]
    )

    history_a_feats = [col for col in schema.names() if 'history_a_' in col]
    df = df.with_columns(
        [pl.col(feat).log1p() for feat in history_a_feats]
    )

    df = df.with_columns([
        pl.col('day_of_week').is_in(['6', '7']).cast(pl.Int8).alias('is_weekend'),
        (np.pi * pl.col('hour') / 12).sin().alias('hour_sin'),
        (np.pi * pl.col('hour') / 12).cos().alias('hour_cos'),
    ])

    df = df.with_columns([
        (pl.col('age_group').cast(str) + "_" + pl.col('gender').cast(str)).alias('age_x_gender')
    ])
    
    df = df.with_columns(
        pl.col('seq').str.split(',').list.len().alias('seq_length')
    )
    
    df = df.with_columns([
        pl.col('seq_length').log1p().alias('log_seq_length'),
        (pl.col('seq_length') < 50).cast(pl.Int8).alias('is_very_short_session'),
        pl.when(pl.col('seq_length') <= 50).then(pl.lit('0-50'))
          .when(pl.col('seq_length') <= 200).then(pl.lit('51-200'))
          .when(pl.col('seq_length') <= 500).then(pl.lit('201-500'))
          .otherwise(pl.lit('500+'))
          .alias('seq_length_binned')
    ])

    if is_train:
        # 학습 시에는 사용자 통계 피처를 계산하여 반환
        user_agg_df = df.group_by('feat_c_2').agg([
            pl.len().alias('user_total_exposure'),
            pl.col('clicked').sum().alias('user_total_clicks'),
            pl.col('inventory_id').n_unique().alias('user_unique_inventories'),
            pl.col('seq_length').mean().alias('user_mean_session_length'),
            pl.col('seq_length').max().alias('user_max_session_length'),
            pl.col('seq_length').std().alias('user_session_length_std'),
        ]).with_columns(
            (pl.col('user_total_clicks') / pl.col('user_total_exposure')).alias('user_ctr')
        ).collect() # Lazy가 아닌 실제 DataFrame으로 변환
        
        df = df.join(user_agg_df.lazy(), on='feat_c_2', how='left')
        print("피처 엔지니어링 완료.")
        return df, user_agg_df
    else:
        # 테스트 시에는 학습 데이터로 만든 통계 피처를 join
        df = df.join(user_agg.lazy(), on='feat_c_2', how='left')
        return df

def objective(trial: optuna.Trial, X: pl.DataFrame, y: pl.Series, categorical_features: list, scale_pos_weight: float) -> float:
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_estimators': 1000,
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

    # Optuna는 scikit-learn wrapper와 함께 사용하는 것이 편리하므로 pandas로 변환
    X_train, X_val, y_train, y_val = train_test_split(X.to_pandas(), y.to_pandas(), test_size=0.2, random_state=42, stratify=y)
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
    print(f"데이터 로드를 시작합니다. 전체 데이터의 {SAMPLE_FRACTION * 100}%를 샘플링합니다...")
    lazy_df = pl.scan_parquet(os.path.join(DATA_PATH, TRAIN_FILE))

    random_index_col = 'random_index'
    sampled_df = lazy_df.with_columns(
        (pl.int_range(0, pl.len()).shuffle().over('clicked')).alias(random_index_col)
    ).filter(
        pl.col(random_index_col) < pl.len() * SAMPLE_FRACTION
    ).drop(random_index_col).collect()

    print(f"데이터 샘플링 완료. 총 {len(sampled_df)}개 행")
    
    processed_df, user_agg_df = feature_engineering(sampled_df.lazy(), is_train=True)
    processed_df = processed_df.collect()

    TARGET = 'clicked'
    
    y = processed_df.get_column(TARGET)
    X = processed_df.drop([TARGET, 'seq'])

    if RUN_OPTUNA:
        # --- Optuna 실행 모드 ---
        categorical_features = []
        for col, dtype in X.schema.items():
            if (dtype == pl.String) or (dtype == pl.Categorical) or (str(dtype) == 'Enum'):
                if str(dtype) != 'Categorical':
                     X = X.with_columns(pl.col(col).cast(pl.Categorical))
                categorical_features.append(col)

        value_counts = y.value_counts()
        count_0 = value_counts.filter(pl.col(TARGET) == 0)['count'][0]
        count_1 = value_counts.filter(pl.col(TARGET) == 1)['count'][0]
        scale_pos_weight = count_0 / count_1
        print(f"scale_pos_weight 계산 완료: {scale_pos_weight:.2f}")
                
        # Optuna에서는 작은 데이터로 여러번 테스트하므로 pandas 변환이 부담스럽지 않음
        study = optuna.create_study(direction='maximize', study_name='lgbm_ctr_tuning')
        func = lambda trial: objective(trial, X, y, categorical_features, scale_pos_weight)
        study.optimize(func, n_trials=N_TRIALS)

        print("\n최적화 완료!")
        print(f"최고 점수 (AUC): {study.best_value:.5f}")
        print("최적 하이퍼파라미터:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
    else:
        # --- 제출 파일 생성 모드 ---
        print("\n최적 파라미터로 최종 모델 학습 및 제출 파일 생성을 시작합니다...")
        
        final_params = BEST_PARAMS.copy()
        final_params['objective'] = 'binary'
        final_params['metric'] = 'auc'
        final_params['verbosity'] = 1 # 학습 과정을 볼 수 있도록 변경
        final_params['boosting_type'] = 'gbdt'
        
        value_counts = y.value_counts()
        count_0 = value_counts.filter(pl.col(TARGET) == 0)['count'][0]
        count_1 = value_counts.filter(pl.col(TARGET) == 1)['count'][0]
        final_params['scale_pos_weight'] = count_0 / count_1
        
        categorical_features = []
        for col, dtype in X.schema.items():
            if (dtype == pl.String) or (dtype == pl.Categorical) or (str(dtype) == 'Enum'):
                if str(dtype) != 'Categorical':
                     X = X.with_columns(pl.col(col).cast(pl.Categorical))
                categorical_features.append(col)
        
        # --- ★★★ 수정된 부분: NumPy 대신 Pandas 데이터프레임 사용 ★★★ ---
        print("LightGBM Dataset 생성을 위해 Pandas로 변환 중...")
        X_pd = X.to_pandas()
        y_pd = y.to_pandas()
        
        lgb_train = lgb.Dataset(X_pd, label=y_pd, feature_name=X_pd.columns.tolist(), categorical_feature=categorical_features, free_raw_data=False)
        
        del X, y, X_pd, y_pd, sampled_df, processed_df
        gc.collect()

        print("최종 모델 학습 중...")
        model = lgb.train(
            params=final_params,
            train_set=lgb_train,
            num_boost_round=1000 # n_estimators 대신 사용
        )
        
        print("테스트 데이터 예측 중...")
        all_preds = []
        all_ids = []
        
        parquet_file = pq.ParquetFile(os.path.join(DATA_PATH, TEST_FILE))
        
        for i, batch in enumerate(parquet_file.iter_batches(batch_size=500_000)):
            print(f"  - 테스트 데이터 Chunk {i+1} 처리 중...")
            chunk_df = pl.from_arrow(batch)
            
            test_processed_lazy = feature_engineering(chunk_df.lazy(), user_agg=user_agg_df, is_train=False)
            test_processed_df = test_processed_lazy.collect()

            test_ids = test_processed_df.get_column('ID')
            X_test = test_processed_df.drop(['ID', 'seq'])
            
            for col, dtype in X_test.schema.items():
                if (dtype == pl.String) or (dtype == pl.Categorical) or (str(dtype) == 'Enum'):
                    if str(dtype) != 'Categorical':
                         X_test = X_test.with_columns(pl.col(col).cast(pl.Categorical))

            X_test_pd = X_test.to_pandas()
            
            # Booster.predict()는 확률 값을 바로 반환
            predictions = model.predict(X_test_pd)
            
            all_ids.extend(test_ids.to_list())
            all_preds.extend(predictions)
            
            del chunk_df, test_processed_df, X_test, X_test_pd
            gc.collect()

        submission_df = pl.DataFrame({
            'ID': all_ids,
            'clicked': all_preds
        })
        
        submission_df.write_csv(SUBMISSION_FILE)
        print(f"\n제출 파일 '{SUBMISSION_FILE}' 생성이 완료되었습니다.")

if __name__ == '__main__':
    main()

