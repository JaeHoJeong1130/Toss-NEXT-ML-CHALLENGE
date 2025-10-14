# 1. 라이브러리 임포트
import os
import duckdb
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import average_precision_score
import gc
import optuna
import joblib

# 2. 메인 실행 함수
def main_duckdb_xgb_final():
    # --- 기본 설정 ---
    DATA_PATH = '/home/jjh/Project/competition/13_toss/data/'
    SUB_PATH = '/home/jjh/Project/competition/13_toss/sub/'
    TRAIN_FILE = os.path.join(DATA_PATH, 'train.parquet')
    TEST_FILE = os.path.join(DATA_PATH, 'test.parquet')
    DUCKDB_FILE = os.path.join(SUB_PATH, 'toss_data.duckdb')
    OPTUNA_PARAMS_FILE = os.path.join(SUB_PATH, 'duckdb_xgb_best_params.pkl')
    OPTUNA_N_TRIALS = 30 # ✅ 옵튜나 30회로 설정
    
    # --- 1단계: DuckDB에 데이터 적재 (샘플링) ---
    con = duckdb.connect(DUCKDB_FILE)
    print("DuckDB에 연결되었습니다.")
    print("학습 데이터 10% 샘플링하여 DuckDB 테이블 생성...")
    con.execute(f"""
        CREATE OR REPLACE TABLE train AS
        SELECT * FROM read_parquet('{TRAIN_FILE}') USING SAMPLE 10%;
    """)
    print("테스트 데이터 DuckDB 테이블 생성...")
    con.execute(f"""
        CREATE OR REPLACE TABLE test AS
        SELECT * FROM read_parquet('{TEST_FILE}');
    """)
    
    # --- 2단계: SQL로 피처 엔지니어링 ---
    print("SQL을 사용하여 피처 엔지니어링 시작...")
    con.execute("""
        CREATE OR REPLACE TABLE agg_features AS
        SELECT inventory_id, AVG(feat_a_1) as feat_a_1_mean, STDDEV(feat_a_1) as feat_a_1_std
        FROM train GROUP BY inventory_id;
    """)
    con.execute("""
        CREATE OR REPLACE TABLE final_train AS
        SELECT t.*, af.feat_a_1_mean, af.feat_a_1_std,
               (LENGTH(t.seq) - LENGTH(REPLACE(t.seq, '_', ''))) + 1 AS seq_length,
               CASE WHEN t.day_of_week IN (5, 6) THEN 1 ELSE 0 END AS is_weekend
        FROM train AS t LEFT JOIN agg_features AS af ON t.inventory_id = af.inventory_id;
    """)
    con.execute("""
        CREATE OR REPLACE TABLE final_test AS
        SELECT t.*, af.feat_a_1_mean, af.feat_a_1_std,
               (LENGTH(t.seq) - LENGTH(REPLACE(t.seq, '_', ''))) + 1 AS seq_length,
               CASE WHEN t.day_of_week IN (5, 6) THEN 1 ELSE 0 END AS is_weekend
        FROM test AS t LEFT JOIN agg_features AS af ON t.inventory_id = af.inventory_id;
    """)
    print("피처 엔지니어링 완료.")

    # --- 3단계: 최종 데이터 추출 및 전처리 ---
    print("최종 데이터를 Pandas DataFrame으로 추출 및 전처리...")
    final_train_pd = con.execute("SELECT * FROM final_train").fetch_df()
    final_test_pd = con.execute("SELECT * FROM final_test").fetch_df()
    con.close()

    for col in ['gender', 'age_group', 'day_of_week', 'hour']:
        final_train_pd[col] = final_train_pd[col].astype('category')
        final_test_pd[col] = final_test_pd[col].astype('category')
    final_train_pd = pd.get_dummies(final_train_pd, columns=['gender', 'age_group', 'day_of_week', 'hour'])
    final_test_pd = pd.get_dummies(final_test_pd, columns=['gender', 'age_group', 'day_of_week', 'hour'])

    train_labels = final_train_pd['clicked']
    test_ids = final_test_pd['ID']
    
    cols_to_drop = ['clicked', 'ID', 'seq', 'inventory_id', 'l_feat_14']
    final_train_pd = final_train_pd.drop(columns=[c for c in cols_to_drop if c in final_train_pd.columns])
    final_test_pd = final_test_pd.drop(columns=[c for c in cols_to_drop if c in final_test_pd.columns])
    
    train_cols = final_train_pd.columns
    final_test_pd = final_test_pd.reindex(columns=train_cols).fillna(0)
    final_train_pd = final_train_pd[final_test_pd.columns]
    final_train_pd.fillna(0, inplace=True)
    
    X = final_train_pd
    y = train_labels
    X_test = final_test_pd

    # --- 4단계: Optuna 하이퍼파라미터 튜닝 ---
    if os.path.exists(OPTUNA_PARAMS_FILE):
        print(f"저장된 최적 파라미터를 불러옵니다: {OPTUNA_PARAMS_FILE}")
        best_params = joblib.load(OPTUNA_PARAMS_FILE)
    else:
        print("Optuna를 사용하여 하이퍼파라미터 튜닝을 시작합니다...")
        
        # Optuna 튜닝용 데이터 분리
        X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        def objective(trial):
            params = {
                'objective': 'binary:logistic', 'eval_metric': 'logloss', 'tree_method': 'hist',
                'n_estimators': 1000,
                'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
                'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
            }
            es_xgb = xgb.callback.EarlyStopping(rounds = 200,metric_name = 'logloss',data_name = 'validation_0',save_best = True,)
            model = xgb.XGBClassifier(**params, callbacks = [es_xgb])
            model.fit(X_train_opt, y_train_opt,
                      eval_set=[(X_val_opt, y_val_opt)],
                      verbose=False)
            preds = model.predict_proba(X_val_opt)[:, 1]
            return average_precision_score(y_val_opt, preds)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=OPTUNA_N_TRIALS)
        best_params = study.best_params
        joblib.dump(best_params, OPTUNA_PARAMS_FILE)

    # --- 5단계: K-Fold 최종 학습 ---
    print("최적 파라미터로 K-Fold 최종 학습을 시작합니다...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    test_predictions = np.zeros(len(X_test))
    
    final_params = best_params
    final_params.update({
        'objective': 'binary:logistic', 'eval_metric': 'logloss',
        'tree_method': 'hist', 'n_estimators': 2000,
    })

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"--- Fold {fold+1}/5 ---")
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        es_xgb = xgb.callback.EarlyStopping(rounds = 200,metric_name = 'logloss',data_name = 'validation_0',save_best = True,)
        model = xgb.XGBClassifier(**final_params, callbacks = [es_xgb])
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  verbose=False)
                  
        fold_preds = model.predict_proba(X_test)[:, 1]
        test_predictions += fold_preds / 5
        del X_train, y_train, X_val, y_val, model; gc.collect()

    # --- 최종 예측 및 제출 ---
    print("최종 예측 및 제출 파일 생성...")
    submission_df = pd.DataFrame({'ID': test_ids, 'clicked': test_predictions})
    submission_df.to_csv(os.path.join(SUB_PATH, 'submission_duckdb_optuna.csv'), index=False)
    
    print("DuckDB 기반 최종 파이프라인이 성공적으로 완료되었습니다!")

if __name__ == '__main__':
    main_duckdb_xgb_final()