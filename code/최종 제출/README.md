# XGBoost CTR 예측 - 학습/추론 분리 가이드

## 📁 파일 구조
```
project/
├── 00_all_in_one.py          # 전처리 코드
├── train.py                   # 학습 코드 (신규)
├── inference.py               # 추론 코드 (신규)
├── Toss/
│   ├── train.parquet         # 원본 학습 데이터
│   ├── test.parquet          # 원본 테스트 데이터
│   ├── sample_submission.csv # 제출 양식
│   ├── _meta/                # 전처리 산출물
│   │   ├── train_enriched_3.parquet
│   │   └── test_enriched_3.parquet
│   ├── new_data/             # 모델 입력 데이터
│   │   ├── new_train_2.parquet
│   │   └── new_test_2.parquet
│   ├── models/               # 학습된 모델 저장 (신규)
│   │   └── xgb_only_v2/
│   │       └── {SEED}/
│   │           ├── fold_1.json
│   │           ├── fold_2.json
│   │           ├── fold_3.json
│   │           ├── fold_4.json
│   │           ├── fold_5.json
│   │           ├── training_metadata.json
│   │           ├── oof_predictions.parquet
│   │           ├── training_log.txt
│   │           └── code_backup/
│   ├── submissions/          # 제출 파일 저장
│   └── log/                  # 시드 관리 로그
```

---

## 🚀 실행 순서

### 1단계: 전처리 (00_all_in_one.py) [중요 : 전처리를 무조건 선행해야 추론 데이터가 생김]
```bash
python 00_all_in_one.py
```

**출력물:**
- `./Toss/_meta/train_enriched_3.parquet`
- `./Toss/_meta/test_enriched_3.parquet`
- `./Toss/new_data/new_train_2.parquet`
- `./Toss/new_data/new_test_2.parquet`

---

### 2단계: 학습 (train.py)
```bash
python train.py
```

**주요 기능:**
- Optuna 하이퍼파라미터 튜닝 (선택)
- K-Fold Cross-Validation (기본 5-fold)
- Temperature Calibration
- 각 fold 모델 자동 저장
- 학습 메타데이터 저장

**출력물:**
- `./Toss/models/xgb_only_v2/{SEED}/fold_*.json` - 각 fold 모델 가중치
- `./Toss/models/xgb_only_v2/{SEED}/training_metadata.json` - 학습 정보
- `./Toss/models/xgb_only_v2/{SEED}/oof_predictions.parquet` - OOF 예측값
- `./Toss/models/xgb_only_v2/{SEED}/training_log.txt` - 학습 로그
- `./Toss/models/xgb_only_v2/{SEED}/code_backup/` - 실행 코드 백업

**학습 로그 예시:**
```
=== Starting 5-Fold Training ===

[XGB][Fold 1/5] scale_pos_weight=2.0000 | tr=800000 va=200000
...
[XGB][Fold 1] AP 0.12345 | WLL 0.45678 | SCORE 0.34567 | best_iter=1234

============================================================
[XGB][OOF] AP 0.12500 | WLL 0.45000 | SCORE 0.35000
============================================================

[CAL] Temperature T=1.234

✓ Training completed successfully!
```

---

### 3단계: 추론 (inference.py)

**중요:** `inference.py` 파일 상단의 `MODEL_DIR` 경로를 실제 학습된 모델 경로로 수정하세요!
```python
# inference.py 파일 수정
MODEL_DIR = Path("./Toss/models/xgb_only_v2/1/")  # 실제 SEED 번호로 변경
```

실행:
```bash
python inference.py
```

**주요 기능:**
- 학습된 모델 자동 로드
- Fold 앙상블 (평균)
- Temperature Calibration 적용
- 제출 파일 생성

**출력물:**
- `./Toss/submissions/submission_{version}_seed{seed}_{score}_{timestamp}.csv`
- `./Toss/submissions/prediction_details_{timestamp}.json`

**추론 로그 예시:**
```
============================================================
XGBoost CTR Prediction - Inference
============================================================

[LOAD] Training metadata loaded
  - Run version: xgb_only_v2
  - Seed: 1
  - Number of folds: 5
  - Temperature: 1.234
  - OOF Score: 0.35000

[LOAD] Loading test data
  - Test shape: (100000, 150)

[INFERENCE] Loading 5 fold models and predicting...
Fold predictions: 100%|████████| 5/5

[ENSEMBLE] Averaged predictions from 5 folds
  - Prediction mean: 0.12345

[CALIBRATION] Applied temperature scaling (T=1.234)
  - Calibrated mean: 0.12340

[SUCCESS] Submission file created!
  - Path: ./Toss/submissions/submission_xgb_only_v2_seed1_0p35000_20250116-123456.csv

✓ Inference completed successfully!
```

---

## ⚙️ 주요 설정

### train.py 설정
```python
CFG = {
    "SEED": SEED,  # 자동 증가 (get_and_bump_seed 함수)
    
    # XGBoost 학습 설정
    "XGB_NUM_BOOST_ROUND": 8000,
    "XGB_ES_ROUNDS": 300,
    "XGB_NFOLDS": 5,
    
    # 데이터 경로
    "META_TRAIN": "./Toss/new_data/new_train_2.parquet",
    "META_TEST":  "./Toss/new_data/new_test_2.parquet",
    
    # Optuna 튜닝 설정
    "OPTUNA_ON": True,
    "OPTUNA_TRIALS": 40,
    "OPTUNA_FOLDS": 3,
    "OPTUNA_NUM_BOOST_ROUND": 5000,
    "OPTUNA_ES": 200,
}

# 빠른 테스트용
SMOKE = False  # True로 설정하면 빠른 실행
```

### inference.py 설정
```python
# 학습된 모델 디렉토리 (필수 수정!)
MODEL_DIR = Path("./Toss/models/xgb_only_v2/1/")

# 데이터 경로
TEST_DATA_PATH = "./Toss/new_data/new_test_2.parquet"
SAMPLE_SUB_PATH = "./Toss/sample_submission.csv"

# 출력 경로
OUTPUT_DIR = Path("./Toss/submissions/")
```

---

## 📊 training_metadata.json 구조

학습 후 생성되는 메타데이터 파일:
```json
{
  "run_version": "xgb_only_v2",
  "seed": 1,
  "n_folds": 5,
  "feature_columns": ["feature1", "feature2", "..."],
  "target_column": "clicked",
  "id_column": "ID",
  "temperature": 1.234,
  "oof_metrics": {
    "ap": 0.12500,
    "wll": 0.45000,
    "score": 0.35000
  },
  "fold_metrics": [
    {
      "fold": 1,
      "ap": 0.12345,
      "wll": 0.45678,
      "score": 0.34567,
      "best_iter": 1234
    },
    ...
  ],
  "fold_models": [
    "fold_1.json",
    "fold_2.json",
    ...
  ],
  "params": { ... },
  "timestamp": "20250116-123456"
}
```

---

## 🔍 체크리스트

### 학습 전 확인사항
- [ ] 전처리 완료 (`new_train_2.parquet`, `new_test_2.parquet` 존재)
- [ ] GPU 사용 가능 (CUDA 설정)
- [ ] 충분한 디스크 공간 (모델 저장용)

### 추론 전 확인사항
- [ ] `train.py` 실행 완료
- [ ] `training_metadata.json` 파일 존재
- [ ] 모든 fold 모델 파일 존재 (`fold_1.json` ~ `fold_5.json`)
- [ ] `inference.py`의 `MODEL_DIR` 경로 수정 완료
- [ ] 테스트 데이터 파일 존재

---

## 🎯 대회 제출용 산출물

### 1. 코드
- ✅ `00_all_in_one.py` - 전처리 코드
- ✅ `train.py` - **학습 코드 (분리됨)**
- ✅ `inference.py` - **추론 코드 (분리됨)**

### 2. 모델 가중치
- ✅ `fold_1.json` ~ `fold_5.json` - XGBoost 모델 파일
- ✅ `training_metadata.json` - 학습 정보 (temperature 포함)

### 3. 제출 파일
- ✅ `submission_*.csv` - 최종 예측 결과

---

## 💡 팁

### 1. Seed 관리
- `train.py`를 실행할 때마다 자동으로 SEED가 증가합니다
- `./Toss/log/SEED_COUNTS_xgb_only_v2.json`에서 현재 시드 확인 가능
- 특정 시드로 고정하려면 `get_and_bump_seed()` 대신 직접 할당

### 2. 빠른 테스트
```python
# train.py에서
SMOKE = True  # 빠른 실행 모드
```

### 3. Optuna 튜닝 스킵
```python
# train.py에서
CFG["OPTUNA_ON"] = False  # 기본 파라미터 사용
```

### 4. 여러 모델 비교
```bash
# 여러 번 학습하여 앙상블
python train.py  # seed=1
python train.py  # seed=2
python train.py  # seed=3

# 각각의 모델로 추론하여 비교
```

### 5. 에러 발생 시
- GPU 메모리 부족: `CFG["XGB_NFOLDS"]` 줄이기
- 학습 시간 오래 걸림: `CFG["OPTUNA_TRIALS"]` 줄이기
- 모델 파일 없음: `MODEL_DIR` 경로 확인

---

## 📧 문제 해결

### Q1. "training_metadata.json not found" 에러
**A:** `train.py`가 정상적으로 완료되었는지 확인하고, `inference.py`의 `MODEL_DIR` 경로가 올바른지 확인하세요.

### Q2. "Feature mismatch" 에러
**A:** 학습과 추론에 동일한 전처리 데이터를 사용하는지 확인하세요.

### Q3. 메모리 부족
**A:** Fold 수를 줄이거나 배치 크기를 조정하세요.

---

## 📝 변경 이력

- **v1.0** (2025-01-16): 학습/추론 코드 분리 초기 버전
  - Optuna 튜닝 지원
  - Temperature calibration
  - 자동 시드 관리
  - 코드 백업 기능