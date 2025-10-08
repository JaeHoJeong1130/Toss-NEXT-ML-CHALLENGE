import os
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import joblib
import json # 피처 목록 저장을 위해 추가
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from tqdm import tqdm
import pyarrow.parquet as pq
import duckdb

sPATH = '/home/jjh/Project/competition/13_toss/sub/'

# --- 1. 설정 (Configuration) ---
# 경로 설정
DATA_PATH = '/home/jjh/Project/competition/13_toss/data/'
TRAIN_FILE = os.path.join(DATA_PATH, 'train.parquet')
TEST_FILE = os.path.join(DATA_PATH, 'test.parquet')
SUBMISSION_FILE = os.path.join(DATA_PATH, 'sample_submission.csv')

# 분할된 데이터, 모델, 결과 저장 경로
SPLIT_DATA_DIR = '/home/jjh/Project/competition/13_toss/splits/'
MODEL_DIR = '/home/jjh/Project/competition/13_toss/models/'
PARAMS_DIR = '/home/jjh/Project/competition/13_toss/params/'
os.makedirs(SPLIT_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PARAMS_DIR, exist_ok=True)

# 모델링 설정
GROUP_COLS = ['age_group', 'inventory_id']
TARGET = 'clicked'
N_SPLITS = 5
N_TRIALS = 5

# --- 2. 데이터 분할 ---
def split_data():
    """train.parquet를 청크 단위로 분할하여 CSV로 저장합니다. 이미 분할된 파일이 있으면 건너뜁니다."""
    print("Step 1: Splitting train data by groups (Memory Optimized)...")

    # [수정] 분할된 데이터가 이미 존재하면 건너뛰기
    if any(f.endswith('.csv') for f in os.listdir(SPLIT_DATA_DIR)):
        print(f"'{SPLIT_DATA_DIR}' already contains split files. Skipping data splitting.")
        return

    written_headers = set()
    parquet_file = pq.ParquetFile(TRAIN_FILE)
    
    print(f"Processing {parquet_file.num_row_groups} row groups from {TRAIN_FILE}...")
    
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

# 나눈 그룹에 해당하는게 없는경우를 위한 글로벌 모델
def train_global_model():
    """DuckDB를 사용하여 메모리 효율적으로 샘플링하고 글로벌 폴백 모델을 학습합니다."""
    global_model_path = os.path.join(MODEL_DIR, 'global_model.pkl')
    
    if os.path.exists(global_model_path):
        print("\nGlobal fallback model already exists. Skipping training.")
        return
        
    print("\nTraining global fallback model using DuckDB for efficient sampling...")
    
    try:
        # DuckDB를 사용해 전체 데이터의 10% (약 100만 건)를 메모리에 직접 로드
        # train.parquet 파일 전체를 읽지 않고 샘플링하므로 메모리 문제 없음
        query = f"SELECT * FROM read_parquet('{TRAIN_FILE}') USING SAMPLE 10 PERCENT (BERNOULLI);"
        df = duckdb.sql(query).df()
        print(f"Successfully sampled {len(df)} rows for the global model.")
            
    except Exception as e:
        print(f"Error during DuckDB sampling: {e}")
        return

    # --- 이하 로직은 기존과 동일 ---
    X = df.drop(columns=[TARGET] + GROUP_COLS, errors='ignore')
    y = df[TARGET]
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    X = X[numeric_features]

    with open(os.path.join(PARAMS_DIR, 'global_features.json'), 'w') as f:
        json.dump(numeric_features, f)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'global_scaler.pkl'))

    params = {
        'objective': 'binary:logistic', 'eval_metric': 'logloss',
        'tree_method': 'gpu_hist', 'gpu_id': 0,
        'n_estimators': 500, 'learning_rate': 0.05,
        'max_depth': 7, 'subsample': 0.8, 'colsample_bytree': 0.8
    }
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_scaled, y)
    
    joblib.dump(model, global_model_path)
    print("Global fallback model training complete.")

# --- 3. 모델 학습 ---
def train_models():
    """분할된 각 데이터에 대해 XGBoost 모델을 학습하고 저장합니다. 이미 학습된 모델은 건너뜁니다."""
    print("\nStep 2: Training a model for each data group (using GPU)...")
    
    split_files = [f for f in os.listdir(SPLIT_DATA_DIR) if f.endswith('.csv')]
    
    for filename in tqdm(split_files, desc="Training Models"):
        group_name = filename.replace('.csv', '')
        
        # [수정] 이미 학습된 모델이 존재하면 건너뛰기
        model_path = os.path.join(MODEL_DIR, f'{group_name}_model.pkl')
        if os.path.exists(model_path):
            continue

        file_path = os.path.join(SPLIT_DATA_DIR, filename)
        
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Could not read {file_path}: {e}")
            continue

        if len(df) < N_SPLITS * 2:
            print(f"Skipping {group_name} due to insufficient data ({len(df)} samples).")
            continue
            
        X = df.drop(columns=[TARGET] + GROUP_COLS, errors='ignore')
        y = df[TARGET]
        numeric_features = X.select_dtypes(include=np.number).columns.tolist()
        X = X[numeric_features]
        
        # [수정] 예측 시 사용할 피처 목록 저장
        features_path = os.path.join(PARAMS_DIR, f'{group_name}_features.json')
        with open(features_path, 'w') as f:
            json.dump(numeric_features, f)

        def objective(trial):
            params = {
                'objective': 'binary:logistic', 'eval_metric': 'logloss', 'verbosity': 0,
                'use_label_encoder': False, 'booster': 'gbtree', 'tree_method': 'gpu_hist', 'gpu_id': 0,
                'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
                'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 9),
                'eta': trial.suggest_float('eta', 0.01, 0.3, log=True),
                'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            }
            cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
            ap_scores = []
            for train_idx, val_idx in cv.split(X, y):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                es_xgb = xgb.callback.EarlyStopping(rounds = 200,metric_name = 'logloss',data_name = 'validation_0',save_best = True,)
                model = xgb.XGBClassifier(**params, callbacks = [es_xgb],)
                model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False
                        #   early_stopping_rounds=EARLY_STOPPING_ROUNDS
                          )
                preds = model.predict_proba(X_val_scaled)[:, 1]
                ap_scores.append(average_precision_score(y_val, preds))
            return np.mean(ap_scores)

        study = optuna.create_study(direction='maximize', study_name=f'xgb_{group_name}')
        study.optimize(objective, n_trials=N_TRIALS)

        best_params = study.best_params
        best_params['tree_method'] = 'gpu_hist'
        best_params['gpu_id'] = 0
        
        print(f"Best AP for {group_name}: {study.best_value}")
        
        joblib.dump(best_params, os.path.join(PARAMS_DIR, f'{group_name}_params.pkl'))
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        final_model = xgb.XGBClassifier(**best_params)
        final_model.fit(X_scaled, y)
        
        joblib.dump(final_model, model_path)
        joblib.dump(scaler, os.path.join(MODEL_DIR, f'{group_name}_scaler.pkl'))

    print("Model training complete.")

# --- 4. 예측 및 제출 파일 생성 ---
def make_predictions():
    """test.parquet에 대해 예측을 수행합니다. 새로운 그룹은 글로벌 모델을 사용합니다."""
    print("\nStep 3: Making predictions on the test set...")
    try:
        test_df = pd.read_parquet(TEST_FILE)
        submission_df = pd.read_csv(SUBMISSION_FILE)
    except Exception as e:
        print(f"Error reading test/submission file: {e}")
        return
    
    # --- [수정] 글로벌 모델 및 관련 파일 미리 로드 ---
    try:
        global_model = joblib.load(os.path.join(MODEL_DIR, 'global_model.pkl'))
        global_scaler = joblib.load(os.path.join(MODEL_DIR, 'global_scaler.pkl'))
        with open(os.path.join(PARAMS_DIR, 'global_features.json'), 'r') as f:
            global_features = json.load(f)
        use_global_model = True
        print("Global fallback model loaded successfully.")
    except FileNotFoundError:
        use_global_model = False
        print("Warning: Global fallback model not found. Defaulting to 0.5 for new groups.")
    # ---------------------------------------------
        
    all_preds = []
    grouped = test_df.groupby(GROUP_COLS)
    
    for name, group in tqdm(grouped, desc="Predicting Test Data"):
        group_name = "_".join(map(str, name))
        model_path = os.path.join(MODEL_DIR, f'{group_name}_model.pkl')
        ids = group['ID']
        
        if os.path.exists(model_path):
            # 기존 로직: 그룹 전용 모델으로 예측
            scaler_path = os.path.join(MODEL_DIR, f'{group_name}_scaler.pkl')
            features_path = os.path.join(PARAMS_DIR, f'{group_name}_features.json')
            
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            
            try:
                with open(features_path, 'r') as f:
                    fit_features = json.load(f)
                X_test = group[fit_features]
                X_test_scaled = scaler.transform(X_test)
                preds = model.predict_proba(X_test_scaled)[:, 1]
                all_preds.append(pd.DataFrame({'ID': ids, TARGET: preds}))

            except FileNotFoundError:
                print(f"Warning: Feature list for {group_name} not found. Skipping prediction for this group.")
                # ID만 있는 빈 데이터프레임을 추가하여 merge시 NaN이 되도록 함
                all_preds.append(pd.DataFrame({'ID': ids}))
        
        else:
            # --- [수정] 새로운 그룹 처리 로직 ---
            if use_global_model:
                # 글로벌 모델로 예측
                X_test_global = group[global_features]
                X_test_global_scaled = global_scaler.transform(X_test_global)
                preds = global_model.predict_proba(X_test_global_scaled)[:, 1]
                all_preds.append(pd.DataFrame({'ID': ids, TARGET: preds}))
            else:
                # 글로벌 모델이 없을 경우의 예비책
                preds = np.full(len(group), 0.5)
                all_preds.append(pd.DataFrame({'ID': ids, TARGET: preds}))
            # ---------------------------------------------

    if not all_preds:
        print("No predictions were made. Submission file will be empty.")
        return

    final_preds_df = pd.concat(all_preds)
    
    # submission 파일과 merge
    submission_df = submission_df[['ID']].merge(final_preds_df, on='ID', how='left')

    # merge 후에도 예측값이 없는 경우(NaN) 0.5로 채우기
    submission_df[TARGET] = submission_df[TARGET].fillna(0.5)
    
    submission_df.to_csv(sPATH + 'submission.csv', index=False)
    print("\nPrediction complete. 'submission.csv' is saved.")

# --- 5. 메인 실행 ---
if __name__ == '__main__':
    split_data()
    train_global_model()
    train_models()
    make_predictions()