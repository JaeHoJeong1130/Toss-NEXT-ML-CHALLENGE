# 파일명: train_kfold_final.py
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
import os
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
    OPTUNA_LGB_TRIALS = 5
    OPTUNA_DEEPFM_TRIALS = 5
    ENSEMBLE_WEIGHTS = {'lgb': 0.6, 'deepfm': 0.4}
    DEEPFM_BATCH_SIZE = 8192
    DEEPFM_MAX_EPOCHS = 20
    DEEPFM_PATIENCE = 2

# --- EarlyStopping 클래스 ---
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience, self.verbose, self.delta, self.path = patience, verbose, delta, path
        self.counter, self.best_score, self.early_stop, self.val_loss_min = 0, None, False, np.Inf
    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose: print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    def save_checkpoint(self, val_loss, model):
        if self.verbose: print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# --- 모델 및 Objective 함수들 (변경 없음) ---
class TabularDataset(Dataset):
    def __init__(self, cat_features, num_features, labels=None):
        self.cat_features, self.num_features, self.labels = cat_features, num_features, labels
    def __len__(self): return len(self.cat_features)
    def __getitem__(self, idx):
        item = {'cat': torch.tensor(self.cat_features[idx], dtype=torch.long), 'num': torch.tensor(self.num_features[idx], dtype=torch.float)}
        if self.labels is not None: item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

class DeepFM(nn.Module):
    def __init__(self, field_dims, num_numerical_features, embedding_dim, dnn_hidden_units, dnn_dropout):
        super().__init__()
        num_fields = len(field_dims)
        self.embeddings = nn.ModuleList([nn.Embedding(num_embeddings, embedding_dim) for num_embeddings in field_dims])
        self.fm_linear = nn.Embedding(sum(field_dims), 1)
        self.offsets = nn.Parameter(torch.tensor((0, *np.cumsum(field_dims)[:-1]), dtype=torch.long), requires_grad=False)
        dnn_input_dim = num_fields * embedding_dim + num_numerical_features
        dnn_layers = []
        for hidden_units in dnn_hidden_units:
            dnn_layers.extend([nn.Linear(dnn_input_dim, hidden_units), nn.BatchNorm1d(hidden_units), nn.ReLU(), nn.Dropout(dnn_dropout)])
            dnn_input_dim = hidden_units
        dnn_layers.append(nn.Linear(dnn_input_dim, 1))
        self.dnn_network = nn.Sequential(*dnn_layers)
    def forward(self, x_cat, x_num):
        embedded_x = torch.stack([emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)], dim=1)
        fm_first_order = self.fm_linear(x_cat + self.offsets).sum(dim=1)
        sum_of_square, square_of_sum = torch.sum(embedded_x, dim=1).pow(2), torch.sum(embedded_x.pow(2), dim=1)
        fm_second_order = 0.5 * torch.sum(sum_of_square - square_of_sum, dim=1, keepdim=True)
        dnn_cat_input = embedded_x.view(x_cat.size(0), -1)
        dnn_input = torch.cat([dnn_cat_input, x_num], dim=1)
        dnn_output = self.dnn_network(dnn_input)
        return torch.sigmoid((fm_first_order + fm_second_order + dnn_output).squeeze(1))

def objective_lgbm(trial, X, y):
    params = { 'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt', 'n_estimators': 10000, 'seed': CFG.RANDOM_STATE, 'n_jobs': -1, 'verbose': -1, 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True), 'num_leaves': trial.suggest_int('num_leaves', 31, 255), 'max_depth': trial.suggest_int('max_depth', 7, 12), 'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0), 'subsample': trial.suggest_float('subsample', 0.6, 1.0), 'reg_alpha': trial.suggest_float('reg_alpha', 1e-2, 10.0, log=True), 'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 10.0, log=True), 'is_unbalance': True, }
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=CFG.RANDOM_STATE, stratify=y)
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='auc', callbacks=[lgb.early_stopping(200, verbose=False)])
    return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

def objective_deepfm(trial, X_train_cat, X_train_num, y_train_values, X_val_cat, X_val_num, y_val_values, field_dims):
    embedding_dim, learning_rate, dnn_dropout = trial.suggest_categorical('embedding_dim', [8, 16, 32]), trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True), trial.suggest_float('dnn_dropout', 0.1, 0.5)
    n_layers = trial.suggest_int('n_layers', 1, 3)
    dnn_hidden_units = [trial.suggest_int(f'n_units_l{i}', 64, 512, log=True) for i in range(n_layers)]
    device = torch.device("cuda")
    model = DeepFM(field_dims, X_train_num.shape[1], embedding_dim, dnn_hidden_units, dnn_dropout).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    train_loader = DataLoader(TabularDataset(X_train_cat, X_train_num, y_train_values), batch_size=CFG.DEEPFM_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TabularDataset(X_val_cat, X_val_num, y_val_values), batch_size=CFG.DEEPFM_BATCH_SIZE)
    model.train()
    for batch in train_loader:
        x_cat, x_num, labels = batch['cat'].to(device), batch['num'].to(device), batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(x_cat, x_num)
        loss = criterion(outputs, labels)
        loss.backward(), optimizer.step()
    model.eval()
    val_preds = []
    with torch.no_grad():
        for batch in val_loader:
            x_cat, x_num = batch['cat'].to(device), batch['num'].to(device)
            val_preds.extend(model(x_cat, x_num).cpu().numpy())
    return roc_auc_score(y_val_values, val_preds)

# --- 메인 실행 로직 ---
def main():
    print("--- 전처리된 데이터로 K-Fold 학습 시작 ---")
    train_df, test_df = pd.read_parquet(CFG.TRAIN_PROCESSED_PATH), pd.read_parquet(CFG.TEST_PROCESSED_PATH)
    
    cat_features = ['gender', 'age_group', 'inventory_id', 'day_of_week', 'hour', 'gender_age_interaction', 'age_inventory_interaction']
    print("범주형 피처 Label Encoding 중...")
    for col in cat_features:
        all_categories = pd.concat([train_df[col].astype('str'), test_df[col].astype('str')]).astype('category').cat.categories
        train_df[col] = pd.Categorical(train_df[col].astype('str'), categories=all_categories).codes
        test_df[col] = pd.Categorical(test_df[col].astype('str'), categories=all_categories).codes

    target_col = 'clicked'
    features = [c for c in train_df.columns if c not in [target_col, 'ID']]
    numerical_features = [c for c in features if c not in cat_features]
    
    print("수치형 피처 스케일링 중...")
    scaler = StandardScaler()
    train_df[numerical_features] = scaler.fit_transform(train_df[numerical_features])
    test_df[numerical_features] = scaler.transform(test_df[numerical_features])

    X, y, X_test = train_df[features], train_df[target_col], test_df[features]
    
    # --- 1. LightGBM 튜닝 및 K-Fold 최종 학습 ---
    print(f"\nOptuna로 LightGBM 튜닝 시작 ({CFG.OPTUNA_LGB_TRIALS}회)...")
    study_lgbm = optuna.create_study(direction='maximize', study_name='lgbm_tuning')
    study_lgbm.optimize(lambda trial: objective_lgbm(trial, X, y), n_trials=CFG.OPTUNA_LGB_TRIALS)
    print(f"\n[LGBM] 최적 AUC: {study_lgbm.best_value}, 최적 파라미터: {study_lgbm.best_params}")
    best_lgbm_params = study_lgbm.best_params
    best_lgbm_params.update({'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt', 'n_estimators': 10000, 'seed': CFG.RANDOM_STATE, 'n_jobs': -1, 'verbose': -1, 'is_unbalance': True})
    
    oof_lgbm_preds = np.zeros(len(X_test))
    skf = StratifiedKFold(n_splits=CFG.N_SPLITS, shuffle=True, random_state=CFG.RANDOM_STATE)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n===== LightGBM Fold {fold+1}/{CFG.N_SPLITS} =====")
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        lgb_model = lgb.LGBMClassifier(**best_lgbm_params)
        lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='auc', callbacks=[lgb.early_stopping(100, verbose=False)])
        oof_lgbm_preds += lgb_model.predict_proba(X_test)[:, 1]
    lgb_preds = oof_lgbm_preds / CFG.N_SPLITS
    print("\nLightGBM K-Fold 예측 완료.")
    del X, y, X_train, y_train, X_val, y_val, lgb_model; gc.collect()

    # --- 2. DeepFM 튜닝 및 K-Fold 최종 학습 ---
    X_cat, X_num, y_values = train_df[cat_features].values, train_df[numerical_features].values, train_df[target_col].values
    X_test_cat, X_test_num = test_df[cat_features].values, test_df[numerical_features].values
    field_dims = [int(pd.concat([train_df[col], test_df[col]]).max()) + 1 for col in cat_features]
    X_train_cat, X_val_cat, X_train_num, X_val_num, y_train_values, y_val_values = train_test_split(X_cat, X_num, y_values, test_size=0.2, random_state=CFG.RANDOM_STATE, stratify=y_values)
    
    print(f"\nOptuna로 DeepFM 튜닝 시작 ({CFG.OPTUNA_DEEPFM_TRIALS}회)...")
    study_deepfm = optuna.create_study(direction='maximize', study_name='deepfm_tuning')
    study_deepfm.optimize(lambda trial: objective_deepfm(trial, X_train_cat, X_train_num, y_train_values, X_val_cat, X_val_num, y_val_values, field_dims), n_trials=CFG.OPTUNA_DEEPFM_TRIALS)
    print(f"\n[DeepFM] 최적 AUC: {study_deepfm.best_value}, 최적 파라미터: {study_deepfm.best_params}")
    best_deepfm_params = study_deepfm.best_params

    oof_deepfm_preds = np.zeros(len(X_test))
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_cat, y_values)):
        print(f"\n===== DeepFM Fold {fold+1}/{CFG.N_SPLITS} =====")
        X_train_cat, X_val_cat = X_cat[train_idx], X_cat[val_idx]
        X_train_num, X_val_num = X_num[train_idx], X_num[val_idx]
        y_train_values, y_val_values = y_values[train_idx], y_values[val_idx]
        
        device = torch.device("cuda")
        model = DeepFM(field_dims, X_num.shape[1], **best_deepfm_params).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=best_deepfm_params['learning_rate'])
        criterion = nn.BCELoss()
        
        train_loader = DataLoader(TabularDataset(X_train_cat, X_train_num, y_train_values), batch_size=CFG.DEEPFM_BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(TabularDataset(X_val_cat, X_val_num, y_val_values), batch_size=CFG.DEEPFM_BATCH_SIZE)
        test_loader = DataLoader(TabularDataset(X_test_cat, X_test_num), batch_size=CFG.DEEPFM_BATCH_SIZE)
        
        es_path = f'deepfm_best_fold_{fold}.pt'
        early_stopping = EarlyStopping(patience=CFG.DEEPFM_PATIENCE, verbose=True, path=es_path)
        
        for epoch in range(CFG.DEEPFM_MAX_EPOCHS):
            model.train()
            train_loss = 0
            for batch in train_loader:
                x_cat, x_num, labels = batch['cat'].to(device), batch['num'].to(device), batch['labels'].to(device)
                optimizer.zero_grad()
                outputs = model(x_cat, x_num)
                loss = criterion(outputs, labels)
                loss.backward(), optimizer.step()
                train_loss += loss.item()
            
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    x_cat, x_num, labels = batch['cat'].to(device), batch['num'].to(device), batch['labels'].to(device)
                    outputs = model(x_cat, x_num)
                    val_loss += criterion(outputs, labels).item()
            
            avg_val_loss = val_loss / len(val_loader)
            print(f"Fold {fold+1} Epoch {epoch+1}: Val Loss: {avg_val_loss:.6f}")
            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop: print("Early stopping"); break
        
        model.load_state_dict(torch.load(es_path))
        model.eval()
        fold_preds = []
        with torch.no_grad():
            for batch in test_loader:
                x_cat, x_num = batch['cat'].to(device), batch['num'].to(device)
                fold_preds.extend(model(x_cat, x_num).cpu().numpy())
        oof_deepfm_preds += np.array(fold_preds)
        os.remove(es_path)

    deepfm_preds = oof_deepfm_preds / CFG.N_SPLITS
    print("\nDeepFM K-Fold 예측 완료.")
    
    # --- 3. 앙상블 및 제출 파일 생성 ---
    print("\n앙상블 및 제출 파일 생성...")
    final_preds = (lgb_preds * CFG.ENSEMBLE_WEIGHTS['lgb']) + (deepfm_preds * CFG.ENSEMBLE_WEIGHTS['deepfm'])
    submission_df = pd.read_csv(CFG.SUBMISSION_PATH)
    submission_df['clicked'] = final_preds
    
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f'submission_kfold_tuned_{current_time}.csv'
    submission_df.to_csv(sPATH + file_name, index=False)
    
    print(f"최종 제출 파일 '{sPATH + file_name}' 생성 완료!")

if __name__ == '__main__':
    main()