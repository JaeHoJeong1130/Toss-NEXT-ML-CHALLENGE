import os
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import joblib
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from tqdm import tqdm
import pyarrow.parquet as pq
import duckdb
from tabpfn import TabPFNClassifier

sPATH = '/home/jjh/Project/competition/13_toss/sub/'

# --- 1. 설정 (Configuration) ---
DATA_PATH = '/home/jjh/Project/competition/13_toss/data/'
TRAIN_FILE = os.path.join(DATA_PATH, 'train.parquet')
TEST_FILE = os.path.join(DATA_PATH, 'test.parquet')
SUBMISSION_FILE = os.path.join(DATA_PATH, 'sample_submission.csv')
SPLIT_DATA_DIR = '/home/jjh/Project/competition/13_toss/splits/'
MODEL_DIR = '/home/jjh/Project/competition/13_toss/models/'
PARAMS_DIR = '/home/jjh/Project/competition/13_toss/params/'
os.makedirs(SPLIT_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PARAMS_DIR, exist_ok=True)

GROUP_COLS = ['age_group', 'inventory_id']
TARGET = 'clicked'
N_SPLITS = 5
N_TRIALS = 5
# EARLY_STOPPING_ROUNDS = 50

# --- 피처 엔지니어링 함수 (seq 피처 추가) ---
def feature_engineer(df, fit_mode=False, artifacts=None):
    """피처 엔지니어링 파이프라인. fit_mode일 때와 아닐 때를 구분."""
    df_processed = df.copy()
    
    # --- [추가] seq 피처 엔지니어링 ---
    # seq 컬럼이 문자열이 아닐 경우를 대비해 str 타입으로 변환하고, 결측치는 빈 문자열로 처리
    df_processed['seq'] = df_processed['seq'].astype(str).fillna('')
    
    # 쉼표(,)를 기준으로 분할하고 숫자로 변환
    sequences = df_processed['seq'].str.split(',').apply(lambda x: [int(i) for i in x if i])

    # 1. 통계 피처 생성
    df_processed['seq_length'] = sequences.apply(len)
    df_processed['seq_unique_count'] = sequences.apply(lambda x: len(set(x)))
    df_processed['seq_mean'] = sequences.apply(lambda x: np.mean(x) if x else 0)
    df_processed['seq_std'] = sequences.apply(lambda x: np.std(x) if x else 0)
    df_processed['seq_max'] = sequences.apply(lambda x: np.max(x) if x else 0)

    # 2. 마지막 N개 행동 피처 생성
    df_processed['seq_last_1'] = sequences.apply(lambda x: x[-1] if len(x) > 0 else -1)
    df_processed['seq_last_2'] = sequences.apply(lambda x: x[-2] if len(x) > 1 else -1)
    # -----------------------------

    ids = df_processed['ID'] if 'ID' in df_processed.columns else None
    
    # 숫자형/범주형 컬럼 정의
    time_cols = ['hour', 'day_of_week']
    categorical_cols = ['gender']
    if 'inventory_id' not in GROUP_COLS:
        categorical_cols.append('inventory_id')
    
    # 숫자형 변환 및 결측치 처리
    for col in time_cols:
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0)
    
    # 시간 피처 변환 (Sine/Cosine)
    for col, period in zip(time_cols, [24, 7]):
        df_processed[f'{col}_sin'] = np.sin(2 * np.pi * df_processed[col] / period)
        df_processed[f'{col}_cos'] = np.cos(2 * np.pi * df_processed[col] / period)
    df_processed = df_processed.drop(columns=time_cols)

    # 원-핫 인코딩
    if fit_mode:
        all_categories = {}
        for col in categorical_cols:
            all_categories[col] = df_processed[col].astype('category').cat.categories.tolist()
        df_processed = pd.get_dummies(df_processed, columns=categorical_cols, dummy_na=False)
        artifacts = {'categories': all_categories}
    else:
        all_categories = artifacts['categories']
        for col, cats in all_categories.items():
            df_processed[col] = pd.Categorical(df_processed[col], categories=cats)
        df_processed = pd.get_dummies(df_processed, columns=categorical_cols, dummy_na=False)

    y = df_processed[TARGET] if TARGET in df_processed.columns else None
    
    # 기존 seq 컬럼은 이제 불필요하므로 제거
    X = df_processed.drop(columns=[TARGET, 'ID', 'seq'] + GROUP_COLS, errors='ignore')
    
    if fit_mode:
        artifacts['columns'] = X.columns.tolist()
        artifacts['numeric_features'] = X.select_dtypes(include=np.number).columns.tolist()
        return X, y, artifacts
    else:
        X = X.reindex(columns=artifacts['columns'], fill_value=0)
        return X, ids

# --- 2. 데이터 분할 ---
def split_data():
    print("Step 1: Splitting train data by groups (Memory Optimized)...")
    if any(f.endswith('.csv') for f in os.listdir(SPLIT_DATA_DIR)):
        print(f"'{SPLIT_DATA_DIR}' already contains split files. Skipping data splitting.")
        return
    written_headers = set()
    parquet_file = pq.ParquetFile(TRAIN_FILE)
    for batch in tqdm(parquet_file.iter_batches(), desc="Splitting Data"):
        chunk_df = batch.to_pandas()
        grouped = chunk_df.groupby(GROUP_COLS)
        for name, group_df in grouped:
            filename = "_".join(map(str, name)) + ".csv"
            save_path = os.path.join(SPLIT_DATA_DIR, filename)
            if filename not in written_headers:
                group_df.to_csv(save_path, index=False, mode='w', header=True)
                written_headers.add(filename)
            else:
                group_df.to_csv(save_path, index=False, mode='a', header=False)
    print("Data splitting complete.")


# --- 글로벌 폴백 모델 학습 (수정된 버전) ---
def train_global_model():
    """DuckDB를 사용하여 메모리 효율적으로 샘플링하고 글로벌 폴백 모델을 학습합니다."""
    global_model_path = os.path.join(MODEL_DIR, 'global_model.pkl')
    if os.path.exists(global_model_path):
        print("\nGlobal fallback model already exists. Skipping training.")
        return
        
    print("\nTraining global fallback model using DuckDB for efficient sampling...")
    try:
        query = f"SELECT * FROM read_parquet('{TRAIN_FILE}') USING SAMPLE 5 PERCENT (BERNOULLI);"
        df = duckdb.sql(query).df()
        print(f"Successfully sampled {len(df)} rows for the global model.")
    except Exception as e:
        print(f"Error during DuckDB sampling: {e}")
        return

    # --- 이하 로직은 기존과 동일 ---
    X, y, artifacts = feature_engineer(df, fit_mode=True)
    
    with open(os.path.join(PARAMS_DIR, 'global_artifacts.json'), 'w') as f:
        json.dump(artifacts, f)
    
    scaler = StandardScaler()
    X[artifacts['numeric_features']] = scaler.fit_transform(X[artifacts['numeric_features']])
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'global_scaler.pkl'))
    
    params = {'objective': 'binary:logistic', 'eval_metric': 'logloss', 'tree_method': 'hist', 'device': 'cuda', 'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 7, 'subsample': 0.8, 'colsample_bytree': 0.8}
    model = xgb.XGBClassifier(**params)
    model.fit(X, y)
    joblib.dump(model, global_model_path)
    print("Global fallback model training complete.")

# --- 3. 그룹별 모델 학습 (하이브리드, DuckDB 샘플링 적용) ---
def train_models():
    """데이터 건수에 따라 TabPFN 또는 XGBoost 모델을 학습합니다. 대용량 그룹은 DuckDB로 샘플링하여 처리합니다."""
    print("\nStep 2: Training models with hybrid strategy (TabPFN/XGBoost)...")
    split_files = [f for f in os.listdir(SPLIT_DATA_DIR) if f.endswith('.csv')]
    
    for filename in tqdm(split_files, desc="Training Models"):
        group_name = filename.replace('.csv', '')
        xgb_path = os.path.join(MODEL_DIR, f'{group_name}_xgb.pkl')
        tabpfn_path = os.path.join(MODEL_DIR, f'{group_name}_tabpfn.pkl')
        if os.path.exists(xgb_path) or os.path.exists(tabpfn_path):
            continue

        file_path = os.path.join(SPLIT_DATA_DIR, filename)
        
        # --- [수정] 대용량 파일 로딩 로직 변경 ---
        MAX_SAMPLES_XGB = 300000 # XGBoost 학습에 사용할 최대 샘플 수 (메모리에 맞게 조절 가능)
        
        # DuckDB로 먼저 전체 행 수를 메모리 소모 없이 확인
        try:
            row_count_query = f"SELECT COUNT(*) FROM read_csv_auto('{file_path}');"
            total_rows = duckdb.sql(row_count_query).fetchone()[0]
        except Exception as e:
            print(f"Could not read file {file_path} with DuckDB. Error: {e}. Skipping.")
            continue

        # TabPFN 조건: 전체 행 수가 10000개 이하일 때
        if total_rows <= 10000:
            df = pd.read_csv(file_path)
            X, y, artifacts = feature_engineer(df, fit_mode=True)
            
            if X.shape[1] <= 100: # 피처 개수 조건 확인
                model = TabPFNClassifier(device='cuda')
                model.fit(X, y, overwrite_warning=True)
                joblib.dump(model, tabpfn_path)
                with open(os.path.join(PARAMS_DIR, f'{group_name}_artifacts.json'), 'w') as f:
                    json.dump(artifacts, f)
                continue # 다음 파일로
        
        # XGBoost 조건: TabPFN 조건을 만족하지 못하는 모든 경우
        print(f"\nGroup {group_name} ({total_rows} samples) -> Preparing for XGBoost.")
        if total_rows > MAX_SAMPLES_XGB:
            print(f"Sampling down to {MAX_SAMPLES_XGB} rows using DuckDB.")
            read_query = f"SELECT * FROM read_csv_auto('{file_path}') USING SAMPLE {MAX_SAMPLES_XGB} ROWS;"
            df = duckdb.sql(read_query).df()
        else:
            df = pd.read_csv(file_path)

        X, y, artifacts = feature_engineer(df, fit_mode=True)
        # ---------------------------------------------------

        if len(df) < N_SPLITS * 2:
            print(f"Skipping group {group_name}: not enough samples for CV.")
            continue
        
        with open(os.path.join(PARAMS_DIR, f'{group_name}_artifacts.json'), 'w') as f:
            json.dump(artifacts, f)

        # (Optuna 로직은 동일)
        def objective(trial):
            params = {'objective': 'binary:logistic', 'eval_metric': 'logloss', 'verbosity': 0, 'booster': 'gbtree', 'tree_method': 'hist', 'device': 'cuda',
                        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
                        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
                        'max_depth': trial.suggest_int('max_depth', 3, 9),
                        'eta': trial.suggest_float('eta', 0.01, 0.3, log=True)}
            cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
            scores = []
            for train_idx, val_idx in cv.split(X, y):
                X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                scaler = StandardScaler()
                X_train[artifacts['numeric_features']] = scaler.fit_transform(X_train[artifacts['numeric_features']])
                X_val[artifacts['numeric_features']] = scaler.transform(X_val[artifacts['numeric_features']])
                es_xgb = xgb.callback.EarlyStopping(rounds = 200,metric_name = 'logloss',data_name = 'validation_0',save_best = True,)
                model = xgb.XGBClassifier(**params, callbacks = [es_xgb],)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                preds = model.predict_proba(X_val)[:, 1]
                scores.append(average_precision_score(y_val, preds))
            return np.mean(scores)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=N_TRIALS)
        best_params = study.best_params
        best_params.update({'tree_method': 'hist', 'device': 'cuda'})
        
        scaler = StandardScaler()
        X[artifacts['numeric_features']] = scaler.fit_transform(X[artifacts['numeric_features']])
        final_model = xgb.XGBClassifier(**best_params)
        final_model.fit(X, y)
        joblib.dump(final_model, xgb_path)
        joblib.dump(scaler, os.path.join(MODEL_DIR, f'{group_name}_scaler.pkl'))
            
    print("Model training complete.")

# --- 4. 예측 및 제출 파일 생성 ---
def make_predictions():
    """일관된 피처 엔지니어링을 적용하여 예측을 수행합니다."""
    print("\nStep 3: Making predictions on the test set...")
    test_df = pd.read_parquet(TEST_FILE)
    submission_df = pd.read_csv(SUBMISSION_FILE)
    
    try:
        global_model = joblib.load(os.path.join(MODEL_DIR, 'global_model.pkl'))
        global_scaler = joblib.load(os.path.join(MODEL_DIR, 'global_scaler.pkl'))
        with open(os.path.join(PARAMS_DIR, 'global_artifacts.json'), 'r') as f:
            global_artifacts = json.load(f)
        use_global_model = True
        print("Global fallback model loaded successfully.")
    except FileNotFoundError:
        use_global_model = False
        print("Warning: Global fallback model not found. Defaulting to 0.5.")
        
    all_preds = []
    grouped = test_df.groupby(GROUP_COLS)
    
    for name, group in tqdm(grouped, desc="Predicting Test Data"):
        group_name = "_".join(map(str, name))
        xgb_path = os.path.join(MODEL_DIR, f'{group_name}_xgb.pkl')
        tabpfn_path = os.path.join(MODEL_DIR, f'{group_name}_tabpfn.pkl')
        artifacts_path = os.path.join(PARAMS_DIR, f'{group_name}_artifacts.json')

        model, scaler, artifacts = None, None, None
        model_type = None

        if os.path.exists(xgb_path):
            model = joblib.load(xgb_path)
            scaler = joblib.load(os.path.join(MODEL_DIR, f'{group_name}_scaler.pkl'))
            with open(artifacts_path, 'r') as f:
                artifacts = json.load(f)
            model_type = 'xgb'
        elif os.path.exists(tabpfn_path):
            model = joblib.load(tabpfn_path)
            with open(artifacts_path, 'r') as f:
                artifacts = json.load(f)
            model_type = 'tabpfn'
        
        if model:
            X_test, ids = feature_engineer(group, fit_mode=False, artifacts=artifacts)
            if model_type == 'xgb':
                X_test[artifacts['numeric_features']] = scaler.transform(X_test[artifacts['numeric_features']])
                preds = model.predict_proba(X_test)[:, 1]
            else: # tabpfn
                preds, _ = model.predict_proba(X_test)
                preds = preds[:, 1]
            all_preds.append(pd.DataFrame({'ID': ids, TARGET: preds}))
        elif use_global_model:
            X_test, ids = feature_engineer(group, fit_mode=False, artifacts=global_artifacts)
            X_test[global_artifacts['numeric_features']] = global_scaler.transform(X_test[global_artifacts['numeric_features']])
            preds = global_model.predict_proba(X_test)[:, 1]
            all_preds.append(pd.DataFrame({'ID': ids, TARGET: preds}))
        else:
            all_preds.append(pd.DataFrame({'ID': group['ID'], TARGET: 0.5}))

    final_preds_df = pd.concat(all_preds)
    submission_df = submission_df[['ID']].merge(final_preds_df, on='ID', how='left')
    submission_df[TARGET] = submission_df[TARGET].fillna(0.05)
    submission_df.to_csv(os.path.join(sPATH, 'submission.csv'), index=False)
    print("\nPrediction complete. 'submission.csv' is saved.")

# --- 5. 메인 실행 ---
if __name__ == '__main__':
    split_data()
    train_global_model()
    train_models()
    make_predictions()