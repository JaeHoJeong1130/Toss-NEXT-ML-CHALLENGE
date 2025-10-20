# 파일명: train_lgbm_only.py
import numpy as np
import pandas as pd
import lightgbm as lgb
import gc
import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

dPATH = '/home/jjh/Project/competition/13_toss/data/'
sPATH = '/home/jjh/Project/competition/13_toss/sub/'

# --- 설정 ---
class CFG:
    TRAIN_PROCESSED_PATH = dPATH + 'train_processed.parquet'
    TEST_PROCESSED_PATH = dPATH + 'test_processed.parquet'
    SUBMISSION_PATH = dPATH + 'sample_submission.csv'
    RANDOM_STATE = 42
    N_SPLITS = 5 # K-Fold 횟수

# --- 메인 실행 로직 ---
def main():
    print("--- 전처리된 데이터로 LGBM K-Fold 학습 시작 ---")
    train_df = pd.read_parquet(CFG.TRAIN_PROCESSED_PATH)
    test_df = pd.read_parquet(CFG.TEST_PROCESSED_PATH)
    
    # LightGBM은 문자열 기반의 상호작용 피처를 직접 처리할 수 있습니다.
    # 따라서 Pandas의 category 코드로 변환하는 과정도 필요 없습니다.
    cat_features = ['gender', 'age_group', 'inventory_id', 'day_of_week', 'hour', 'gender_age_interaction', 'age_inventory_interaction']
    for col in cat_features:
        train_df[col] = train_df[col].astype('category')
        test_df[col] = test_df[col].astype('category')

    target_col = 'clicked'
    features = [c for c in train_df.columns if c not in [target_col, 'ID']]
    numerical_features = [c for c in features if c not in cat_features]
    
    # 수치형 피처 스케일링은 트리 모델에 필수는 아니지만, 유지해도 괜찮습니다.
    print("수치형 피처 스케일링 중...")
    scaler = StandardScaler()
    train_df[numerical_features] = scaler.fit_transform(train_df[numerical_features])
    test_df[numerical_features] = scaler.transform(test_df[numerical_features])

    X = train_df[features]
    y = train_df[target_col]
    X_test = test_df[features]
    
    # --- LightGBM K-Fold 최종 학습 ---
    # Optuna로 이전에 찾은 최적의 하이퍼파라미터
    best_lgbm_params = {
        'learning_rate': 0.0100974348059352, 
        'num_leaves': 255, 
        'max_depth': 12, 
        'colsample_bytree': 0.6037808406200595, 
        'subsample': 0.8323674671263116, 
        'reg_alpha': 7.907057909100102, 
        'reg_lambda': 7.962388810131502
    }
    print("[LGBM] 저장된 최적 파라미터를 사용합니다.")
    best_lgbm_params.update({'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt', 'n_estimators': 10000, 'seed': CFG.RANDOM_STATE, 'n_jobs': -1, 'verbose': -1, 'is_unbalance': True})
    
    oof_lgbm_preds = np.zeros(len(X_test))
    skf = StratifiedKFold(n_splits=CFG.N_SPLITS, shuffle=True, random_state=CFG.RANDOM_STATE)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n===== LightGBM Fold {fold+1}/{CFG.N_SPLITS} =====")
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        # LightGBM에 범주형 피처를 명시적으로 알려줌
        lgb_model = lgb.LGBMClassifier(**best_lgbm_params)
        lgb_model.fit(X_train, y_train, 
                      eval_set=[(X_val, y_val)], 
                      eval_metric='auc', 
                      callbacks=[lgb.early_stopping(100, verbose=False)],
                      categorical_feature=cat_features)
        
        oof_lgbm_preds += lgb_model.predict_proba(X_test)[:, 1]
    
    final_preds = oof_lgbm_preds / CFG.N_SPLITS
    print("\nLightGBM K-Fold 예측 완료.")
    
    # --- 최종 제출 파일 생성 ---
    print("\n제출 파일 생성 중...")
    submission_df = pd.read_csv(CFG.SUBMISSION_PATH)
    submission_df['clicked'] = final_preds
    
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f'submission_lgbm_kfold_{current_time}.csv'
    submission_df.to_csv(sPATH + file_name, index=False)
    
    print(f"최종 제출 파일 '{sPATH + file_name}' 생성 완료!")

if __name__ == '__main__':
    main()