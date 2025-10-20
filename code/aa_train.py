import polars as pl
import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import gc
import datetime
import optuna
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

dPATH = '/home/jjh/Project/competition/13_toss/data/'
sPATH = '/home/jjh/Project/competition/13_toss/sub/'

# --- 설정 ---
class CFG:
    TRAIN_PROCESSED_PATH = dPATH + 'train_processed.parquet'
    TEST_PROCESSED_PATH = dPATH + 'test_processed.parquet'
    SUBMISSION_PATH = dPATH + 'sample_submission.csv'
    RANDOM_STATE = 42
    N_SPLITS = 5
    OPTUNA_LGB_TRIALS = 20
    OPTUNA_DEEPFM_TRIALS = 15
    ENSEMBLE_WEIGHTS = {'lgb': 0.6, 'deepfm': 0.4}
    DEEPFM_BATCH_SIZE = 8192
    DEEPFM_EPOCHS = 3 # ✨ 오류 해결: 설정값 복원

# ✨ --- DeepFM 모델 업그레이드 --- ✨
class TabularDataset(Dataset):
    def __init__(self, cat_features, num_features, labels=None):
        self.cat_features = cat_features
        self.num_features = num_features
        self.labels = labels
    def __len__(self): return len(self.cat_features)
    def __getitem__(self, idx):
        item = {
            'cat': torch.tensor(self.cat_features[idx], dtype=torch.long),
            'num': torch.tensor(self.num_features[idx], dtype=torch.float)
        }
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

class DeepFM(nn.Module):
    def __init__(self, field_dims, num_numerical_features, embedding_dim, dnn_hidden_units, dnn_dropout):
        super().__init__()
        num_fields = len(field_dims)
        self.embeddings = nn.ModuleList([nn.Embedding(num_embeddings, embedding_dim) for num_embeddings in field_dims])
        self.fm_linear = nn.Embedding(sum(field_dims), 1)
        self.offsets = nn.Parameter(torch.tensor((0, *np.cumsum(field_dims)[:-1]), dtype=torch.long), requires_grad=False)
        
        # DNN 파트 입력 차원 변경 (수치형 피처 추가)
        dnn_input_dim = num_fields * embedding_dim + num_numerical_features
        
        dnn_layers = []
        for hidden_units in dnn_hidden_units:
            dnn_layers.append(nn.Linear(dnn_input_dim, hidden_units))
            dnn_layers.append(nn.BatchNorm1d(hidden_units))
            dnn_layers.append(nn.ReLU())
            dnn_layers.append(nn.Dropout(dnn_dropout))
            dnn_input_dim = hidden_units
        dnn_layers.append(nn.Linear(dnn_input_dim, 1))
        self.dnn_network = nn.Sequential(*dnn_layers)

    def forward(self, x_cat, x_num):
        embedded_x = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        embedded_x = torch.stack(embedded_x, dim=1)
        
        fm_first_order = self.fm_linear(x_cat + self.offsets).sum(dim=1)
        sum_of_square = torch.sum(embedded_x, dim=1).pow(2)
        square_of_sum = torch.sum(embedded_x.pow(2), dim=1)
        fm_second_order = 0.5 * torch.sum(sum_of_square - square_of_sum, dim=1, keepdim=True)
        
        # DNN 입력에 수치형 피처 결합
        dnn_cat_input = embedded_x.view(x_cat.size(0), -1)
        dnn_input = torch.cat([dnn_cat_input, x_num], dim=1)
        
        dnn_output = self.dnn_network(dnn_input)
        output = fm_first_order + fm_second_order + dnn_output
        return torch.sigmoid(output.squeeze(1))

# Optuna Objective 함수들
def objective_lgbm(trial, X, y):
    params = {
        'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt', 'n_estimators': 10000, 
        'seed': CFG.RANDOM_STATE, 'n_jobs': -1, 'verbose': -1,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 255),
        'max_depth': trial.suggest_int('max_depth', 7, 12),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-2, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 10.0, log=True),
        'is_unbalance': True,
    }
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=CFG.RANDOM_STATE, stratify=y)
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='auc', callbacks=[lgb.early_stopping(100, verbose=False)])
    preds = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, preds)

def objective_deepfm(trial, X_train_cat, X_train_num, y_train_values, X_val_cat, X_val_num, y_val_values, field_dims):
    embedding_dim = trial.suggest_categorical('embedding_dim', [8, 16, 32])
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    dnn_dropout = trial.suggest_float('dnn_dropout', 0.1, 0.5)
    n_layers = trial.suggest_int('n_layers', 1, 3)
    dnn_hidden_units = [trial.suggest_int(f'n_units_l{i}', 64, 512, log=True) for i in range(n_layers)]
    
    device = torch.device("cuda")
    model = DeepFM(field_dims, X_train_num.shape[1], embedding_dim, dnn_hidden_units, dnn_dropout).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    
    train_dataset = TabularDataset(X_train_cat, X_train_num, y_train_values)
    train_loader = DataLoader(train_dataset, batch_size=CFG.DEEPFM_BATCH_SIZE, shuffle=True)
    val_dataset = TabularDataset(X_val_cat, X_val_num, y_val_values)
    val_loader = DataLoader(val_dataset, batch_size=CFG.DEEPFM_BATCH_SIZE, shuffle=False)
    
    model.train()
    for batch in train_loader:
        x_cat, x_num, labels = batch['cat'].to(device), batch['num'].to(device), batch['labels'].to(device)
        features_tensor, labels = batch['features'].to(device), batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(x_cat, x_num)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
    model.eval()
    val_preds = []
    with torch.no_grad():
        for batch in val_loader:
            x_cat, x_num = batch['cat'].to(device), batch['num'].to(device)
            features_tensor = batch['features'].to(device)
            outputs = model(x_cat, x_num)
            val_preds.extend(outputs.cpu().numpy())
            
    return roc_auc_score(y_val_values, val_preds)

# --- 메인 실행 로직 ---
def main():
    print("--- 전처리된 데이터로 학습 시작 ---")
    train_df = pl.read_parquet(CFG.TRAIN_PROCESSED_PATH).to_pandas()
    test_df = pl.read_parquet(CFG.TEST_PROCESSED_PATH).to_pandas()
    
    categorical_cols_new = ['gender_age_interaction', 'age_inventory_interaction']
    for col in categorical_cols_new:
        all_categories = pd.concat([train_df[col].astype('str'), test_df[col].astype('str')]).astype('category').cat.categories
        train_df[col] = pd.Categorical(train_df[col], categories=all_categories).codes
        test_df[col] = pd.Categorical(test_df[col], categories=all_categories).codes

    target_col = 'clicked'
    features = [c for c in train_df.columns if c not in [target_col, 'ID']]
    X = train_df[features]
    y = train_df[target_col]
    X_test = test_df[features]
    
    # --- 1. LightGBM 튜닝 및 최종 학습 ---
    
    # --- 💡 이미 찾은 최적 파라미터를 사용 ---
    best_lgbm_params = {'learning_rate': 0.010004570880768284, 'num_leaves': 174, 'max_depth': 12, 'colsample_bytree': 0.6070601676678545, 'subsample': 0.6232114462258108, 'reg_alpha': 0.08879310792905541, 'reg_lambda': 0.010320658054049534}
    print("[LGBM] 저장된 최적 파라미터를 사용합니다.")

    # --- 💡 다시 튜닝하려면 아래 코드 ---
    # print(f"\nOptuna로 LightGBM 튜닝 시작 ({CFG.OPTUNA_LGB_TRIALS}회)...")
    # study_lgbm = optuna.create_study(direction='maximize', study_name='lgbm_tuning')
    # study_lgbm.optimize(lambda trial: objective_lgbm(trial, X, y), n_trials=CFG.OPTUNA_LGB_TRIALS)
    # print(f"\n[LGBM] 최적 AUC: {study_lgbm.best_value}, 최적 파라미터: {study_lgbm.best_params}")
    # best_lgbm_params = study_lgbm.best_params
    
    best_lgbm_params.update({'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt', 'n_estimators': 10000, 'seed': CFG.RANDOM_STATE, 'n_jobs': -1, 'verbose': -1, 'is_unbalance': True})

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=CFG.RANDOM_STATE, stratify=y)
    final_lgb_model = lgb.LGBMClassifier(**best_lgbm_params)
    final_lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='auc', callbacks=[lgb.early_stopping(100, verbose=True)])
    lgb_preds = final_lgb_model.predict_proba(X_test)[:, 1]
    print("LightGBM 최종 예측 완료.")
    del final_lgb_model, X, y, X_train, y_train, X_val, y_val
    gc.collect()

    # --- 2. DeepFM 튜닝 및 최종 학습 ---
    deepfm_cat_features = ['gender', 'age_group', 'inventory_id', 'day_of_week', 'hour']
    X_cat = train_df[deepfm_cat_features].values
    y_values = train_df[target_col].values
    X_test_cat = test_df[deepfm_cat_features].values
    field_dims = [int(np.max(X_cat[:, i])) + 1 for i in range(X_cat.shape[1])]
    
    X_train_cat, X_val_cat, y_train_values, y_val_values = train_test_split(X_cat, y_values, test_size=0.2, random_state=CFG.RANDOM_STATE, stratify=y_values)
    
    print(f"\nOptuna로 DeepFM 튜닝 시작 ({CFG.OPTUNA_DEEPFM_TRIALS}회)...")
    study_deepfm = optuna.create_study(direction='maximize', study_name='deepfm_tuning')
    study_deepfm.optimize(lambda trial: objective_deepfm(trial, X_train_cat, y_train_values, X_val_cat, y_val_values, field_dims), n_trials=CFG.OPTUNA_DEEPFM_TRIALS)

    print(f"\n[DeepFM] 최적 AUC: {study_deepfm.best_value}, 최적 파라미터: {study_deepfm.best_params}")

    best_deepfm_params = study_deepfm.best_params
    best_embedding_dim = best_deepfm_params['embedding_dim']
    best_learning_rate = best_deepfm_params['learning_rate']
    best_dnn_dropout = best_deepfm_params['dnn_dropout']
    best_n_layers = best_deepfm_params['n_layers']
    best_dnn_hidden_units = [best_deepfm_params[f'n_units_l{i}'] for i in range(best_n_layers)]
    
    device = torch.device("cuda")
    final_deepfm_model = DeepFM(field_dims, best_embedding_dim, best_dnn_hidden_units, best_dnn_dropout).to(device)
    optimizer = optim.AdamW(final_deepfm_model.parameters(), lr=best_learning_rate)
    criterion = nn.BCELoss()
    
    train_dataset = TabularDataset(X_cat, y_values) # 전체 데이터로 최종 학습
    train_loader = DataLoader(train_dataset, batch_size=CFG.DEEPFM_BATCH_SIZE, shuffle=True, num_workers=4)
    test_dataset = TabularDataset(X_test_cat)
    test_loader = DataLoader(test_dataset, batch_size=CFG.DEEPFM_BATCH_SIZE, shuffle=False, num_workers=4)

    print("\n최적의 파라미터로 최종 DeepFM 모델 학습...")
    final_deepfm_model.train()
    for epoch in range(CFG.DEEPFM_EPOCHS):
        for batch in tqdm(train_loader, desc=f"Final DeepFM Epoch {epoch+1}/{CFG.DEEPFM_EPOCHS}"):
            features_tensor, labels = batch['features'].to(device), batch['labels'].to(device)
            optimizer.zero_grad()
            outputs = final_deepfm_model(features_tensor)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
    final_deepfm_model.eval()
    deepfm_preds = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="DeepFM 최종 예측 중"):
            features_tensor = batch['features'].to(device)
            outputs = final_deepfm_model(features_tensor)
            deepfm_preds.extend(outputs.cpu().numpy())
    deepfm_preds = np.array(deepfm_preds)
    print("DeepFM 최종 예측 완료.")
    
    # --- 3. 앙상블 및 제출 파일 생성 ---
    print("\n앙상블 및 제출 파일 생성...")
    final_preds = (lgb_preds * CFG.ENSEMBLE_WEIGHTS['lgb']) + (deepfm_preds * CFG.ENSEMBLE_WEIGHTS['deepfm'])
    submission_df = pd.read_csv(CFG.SUBMISSION_PATH)
    submission_df['clicked'] = final_preds
    
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f'submission_tuned_{current_time}.csv'
    submission_df.to_csv(sPATH + file_name, index=False)
    
    print(f"최종 제출 파일 '{sPATH + file_name}' 생성 완료!")

if __name__ == '__main__':
    main()