# -*- coding: utf-8 -*-
"""
Toss NEXT ML Challenge - Final Stable Version

- Environment: RAPIDS 24.08 (conda/mamba)
- Features:
    1. NVTabular for fast, out-of-core preprocessing.
    2. Optuna for hyperparameter optimization (single validation split).
    3. XGBoost's scale_pos_weight for imbalance handling.
    4. Final model training and submission file generation.
"""

# =========== 진단 코드 시작 ===========
import sys
import os

print("--- Python Interpreter & Path Diagnosis ---")
print(f"Python Executable: {sys.executable}")
print("\nPython Path (sys.path):")
for path in sys.path:
    print(f"  - {path}")
print("\n--- NumPy Diagnosis ---")
try:
    import numpy
    print(f"NumPy Version: {numpy.__version__}")
    print(f"NumPy File Path: {numpy.__file__}")
except ImportError as e:
    print(f"Failed to import NumPy: {e}")
except Exception as e:
    print(f"An error occurred during NumPy import: {e}")
print("--- Diagnosis End ---\n\n")
# =========== 진단 코드 끝 ===========


# --- 기존 코드 시작 ---

# --- 1. Imports ---
import os
import gc
import time
import shutil
import warnings
import glob

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import cupy as cp
import cudf
import nvtabular as nvt
from nvtabular import ops
from merlin.io import Dataset
import xgboost as xgb
from sklearn.metrics import average_precision_score
import optuna

print("✅ All libraries imported successfully")

# --- 2. Configuration ---
class CONFIG:
    BASE_PATH = '/home/jjh/Project/competition/13_toss/'
    DATA_PATH = os.path.join(BASE_PATH, 'data/')
    SUB_PATH = os.path.join(BASE_PATH, 'sub/')
    TRAIN_PATH = os.path.join(DATA_PATH, 'train.parquet')
    TEST_PATH = os.path.join(DATA_PATH, 'test.parquet')
    OUTPUT_DIR = os.path.join(DATA_PATH, 'nvt_processed_final')
    OPTUNA_DB_PATH = os.path.join(BASE_PATH, 'optuna_study.db')
    FORCE_REPROCESS = False
    RANDOM_STATE = 42
    N_TRIALS = 50

os.makedirs(CONFIG.SUB_PATH, exist_ok=True)
print("📋 Configuration and paths set.")

# --- 3. Utility Functions ---
def clear_gpu_memory():
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()

def calculate_competition_score(y_true, y_pred, eps=1e-15):
    y_true_np = cp.asnumpy(y_true) if isinstance(y_true, cp.ndarray) else y_true
    y_pred_np = cp.asnumpy(y_pred) if isinstance(y_pred, cp.ndarray) else y_pred
    ap = average_precision_score(y_true_np, y_pred_np)
    y_pred_clipped = np.clip(y_pred_np, eps, 1 - eps)
    mask_0, mask_1 = (y_true_np == 0), (y_true_np == 1)
    ll_0 = -np.mean(np.log(1 - y_pred_clipped[mask_0])) if mask_0.sum() > 0 else 0
    ll_1 = -np.mean(np.log(y_pred_clipped[mask_1])) if mask_1.sum() > 0 else 0
    wll = 0.5 * ll_0 + 0.5 * ll_1
    return 0.5 * ap + 0.5 * (1 / (1 + wll))

# --- 4. Data Processing (NVTabular) ---
def create_workflow():
    true_categorical = ['gender', 'age_group', 'inventory_id', 'day_of_week', 'hour']
    all_continuous = (
        [f'feat_a_{i}' for i in range(1, 19)] + [f'feat_b_{i}' for i in range(1, 7)] +
        [f'feat_c_{i}' for i in range(1, 9)] + [f'feat_d_{i}' for i in range(1, 7)] +
        [f'feat_e_{i}' for i in range(1, 11)] + [f'history_a_{i}' for i in range(1, 8)] +
        [f'history_b_{i}' for i in range(1, 31)] + [f'l_feat_{i}' for i in range(1, 28)]
    )
    cat_features = true_categorical >> ops.Categorify(freq_threshold=0, max_size=50000)
    cont_features = all_continuous >> ops.FillMissing(fill_val=0)
    workflow = nvt.Workflow(cat_features + cont_features + ['clicked'])
    return workflow

def process_data(file_path, is_train=True):
    print(f"\n🚀 Processing {'Train' if is_train else 'Test'} Data...")
    
    if not is_train:
        workflow = nvt.Workflow.load(os.path.join(CONFIG.OUTPUT_DIR, 'workflow'))
        test_dataset = Dataset(file_path, engine='parquet', part_size='32MB', strings_to_categorical=True)
        processed_test_dir = os.path.join(CONFIG.DATA_PATH, 'nvt_processed_test')
        if os.path.exists(processed_test_dir):
            shutil.rmtree(processed_test_dir)
        workflow.transform(test_dataset).to_parquet(output_path=processed_test_dir)
        print(f"   ✅ Test data processed and saved to {processed_test_dir}")
        return processed_test_dir

    if os.path.exists(CONFIG.OUTPUT_DIR) and not CONFIG.FORCE_REPROCESS:
        print(f"✅ Using existing processed data from {CONFIG.OUTPUT_DIR}")
        return CONFIG.OUTPUT_DIR

    if os.path.exists(CONFIG.OUTPUT_DIR):
        shutil.rmtree(CONFIG.OUTPUT_DIR)
    
    dataset = Dataset(file_path, engine='parquet', part_size='32MB', strings_to_categorical=True)
    workflow = create_workflow()
    workflow.fit(dataset)
    
    workflow.transform(dataset).to_parquet(output_path=CONFIG.OUTPUT_DIR)
    workflow.save(os.path.join(CONFIG.OUTPUT_DIR, 'workflow'))
    print(f"   ✅ Train data processed and saved to {CONFIG.OUTPUT_DIR}")
    return CONFIG.OUTPUT_DIR

# --- 5. Optuna Objective Function ---
def objective(trial, X_train, y_train, X_val, y_val):
    scale_pos_weight = float(cp.sum(y_train == 0) / cp.sum(y_train == 1))
    params = {
        'objective': 'binary:logistic', 'tree_method': 'gpu_hist', 'gpu_id': 0, 'verbosity': 0,
        'seed': CONFIG.RANDOM_STATE, 'scale_pos_weight': scale_pos_weight,
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 100),
    }
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    model = xgb.train(params, dtrain, num_boost_round=1000, evals=[(dval, 'val')],
                      early_stopping_rounds=50, verbose_eval=False)
    y_pred = model.predict(dval, iteration_range=(0, model.best_iteration))
    score = calculate_competition_score(y_val, y_pred)
    del dtrain, dval, model; clear_gpu_memory()
    return score

# --- 6. Final Model Training and Prediction ---
def train_final_model_and_predict(best_params, all_files, test_data_path):
    print("\n" + "="*70 + "\n🔥 Training Final Model and Generating Submission\n" + "="*70)
    
    print("   Loading full training data for final model...")
    full_train_dataset = Dataset(all_files, engine='parquet', part_size='256MB', strings_to_categorical=True)
    full_train_gdf = full_train_dataset.to_ddf().compute()
    y_full, X_full = full_train_gdf['clicked'].values, full_train_gdf.drop('clicked', axis=1).astype('float32').values
    dtrain_full = xgb.DMatrix(X_full, label=y_full)
    del full_train_gdf, X_full, y_full; gc.collect()
    print("   DMatrix creation complete.")

    final_model = xgb.train(best_params, dtrain_full, num_boost_round=1500, verbose_eval=100)
    
    processed_test_dir = process_data(test_data_path, is_train=False)
    test_gdf = cudf.read_parquet(processed_test_dir)
    X_test = test_gdf.astype('float32').values
    dtest = xgb.DMatrix(X_test)
    predictions = final_model.predict(dtest)
    
    sub_df = pd.read_csv(os.path.join(CONFIG.DATA_PATH, 'sample_submission.csv'))
    sub_df['clicked'] = predictions
    sub_path = os.path.join(CONFIG.SUB_PATH, 'submission.csv')
    sub_df.to_csv(sub_path, index=False)
    print(f"   ✅ Submission file saved to: {sub_path}")

# --- 7. Main Execution ---
def main():
    processed_train_dir = process_data(CONFIG.TRAIN_PATH, is_train=True)
    
    all_files = sorted(glob.glob(os.path.join(processed_train_dir, "*.parquet")))
    # 마지막 2개 파티션을 검증용으로, 나머지를 학습용으로 사용
    val_files, train_files = all_files[-2:], all_files[:-2]
    
    print(f"   Train files: {len(train_files)}, Validation files: {len(val_files)}")
    
    train_dataset = Dataset(train_files, engine='parquet', part_size='256MB', strings_to_categorical=True)
    train_gdf = train_dataset.to_ddf().compute()
    y_train, X_train = train_gdf['clicked'].values, train_gdf.drop('clicked', axis=1).astype('float32').values
    del train_gdf; gc.collect()

    val_dataset = Dataset(val_files, engine='parquet', part_size='256MB', strings_to_categorical=True)
    val_gdf = val_dataset.to_ddf().compute()
    y_val, X_val = val_gdf['clicked'].values, val_gdf.drop('clicked', axis=1).astype('float32').values
    del val_gdf; gc.collect(); clear_gpu_memory()
    
    storage = f"sqlite:///{CONFIG.OPTUNA_DB_PATH}"
    study = optuna.create_study(study_name="toss-ctr-optimization-latest", storage=storage,
                                load_if_exists=True, direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val),
                   n_trials=CONFIG.N_TRIALS, gc_after_trial=True)
    
    best_params = {'objective': 'binary:logistic', 'tree_method': 'gpu_hist', 'gpu_id': 0, 'verbosity': 0, 'seed': CONFIG.RANDOM_STATE}
    best_params.update(study.best_params)
    
    # Optuna에 사용했던 y_train, y_val을 합쳐서 전체 학습 데이터의 scale_pos_weight를 계산
    y_full = cp.concatenate([y_train, y_val])
    scale_pos_weight = float(cp.sum(y_full == 0) / cp.sum(y_full == 1))
    best_params['scale_pos_weight'] = scale_pos_weight
    del y_train, X_train, y_val, X_val, y_full; gc.collect()
    
    train_final_model_and_predict(best_params, all_files, CONFIG.TEST_PATH)

if __name__ == "__main__":
    main()