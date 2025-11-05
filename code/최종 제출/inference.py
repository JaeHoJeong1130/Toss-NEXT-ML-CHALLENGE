# -*- coding: utf-8 -*-
"""
Inference script: Load trained XGBoost models and generate predictions
Requires:
  - Training metadata (training_metadata.json)
  - Fold models (fold_*.json)
  - Test data (new_test_2.parquet)
Outputs:
  - Submission file (submission.csv)
"""
import os, gc, warnings, datetime, json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import xgboost as xgb

# ---------------- Config ----------------
# Path to trained model directory (modify this to point to your trained models)
MODEL_DIR = Path("./Toss/models/xgb_only_v2/9/")  # Example: change to your actual path

# Data paths
TEST_DATA_PATH = "./Toss/new_data/new_test_2.parquet"
SAMPLE_SUB_PATH = "./Toss/sample_submission.csv"

# Output path
OUTPUT_DIR = Path("./Toss/submissions/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- Utils ----------------
def free_mem():
    gc.collect()

def _clip01(p):
    return np.clip(p, 1e-6, 1-1e-6)

def _logit(p):
    p = _clip01(p)
    return np.log(p/(1-p))

def _sigmoid(z):
    return 1/(1+np.exp(-z))

def apply_temperature(p, T):
    """Apply temperature scaling to predictions"""
    return _sigmoid(_logit(p) / T)

# ---------------- Main Inference ----------------
def main():
    print("="*60)
    print("XGBoost CTR Prediction - Inference")
    print("="*60)
    
    # 1. Load training metadata
    metadata_path = MODEL_DIR / "training_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Training metadata not found: {metadata_path}")
    
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    print(f"\n[LOAD] Training metadata loaded from: {metadata_path}")
    print(f"  - Run version: {metadata['run_version']}")
    print(f"  - Seed: {metadata['seed']}")
    print(f"  - Number of folds: {metadata['n_folds']}")
    print(f"  - Temperature: {metadata['temperature']:.3f}")
    print(f"  - OOF Score: {metadata['oof_metrics']['score']:.5f}")
    
    feature_cols = metadata["feature_columns"]
    temperature = metadata["temperature"]
    fold_models = metadata["fold_models"]
    ID_COL = metadata["id_column"]
    
    # 2. Load test data
    print(f"\n[LOAD] Loading test data from: {TEST_DATA_PATH}")
    test_df = pd.read_parquet(TEST_DATA_PATH, engine="pyarrow")
    print(f"  - Test shape: {test_df.shape}")
    
    # Ensure ID column
    if ID_COL in test_df.columns:
        test_df[ID_COL] = test_df[ID_COL].astype(str)
    else:
        test_df[ID_COL] = pd.RangeIndex(len(test_df)).astype(str)
    
    ids_test = test_df[ID_COL].values
    
    # Prepare features
    # Check if all feature columns exist in test data
    missing_cols = [c for c in feature_cols if c not in test_df.columns]
    if missing_cols:
        print(f"[WARNING] Missing columns in test data: {missing_cols}")
        print("[INFO] Creating missing columns with NaN values")
        for col in missing_cols:
            test_df[col] = np.nan
    
    # Replace inf and cast to float32
    test_df[feature_cols] = test_df[feature_cols].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    X_test = test_df[feature_cols].values
    
    print(f"  - Feature shape: {X_test.shape}")
    print(f"  - Number of features: {len(feature_cols)}")
    
    del test_df
    free_mem()
    
    # 3. Load models and make predictions
    print(f"\n[INFERENCE] Loading {len(fold_models)} fold models and predicting...")
    
    test_preds = np.zeros((len(X_test), len(fold_models)), dtype=np.float32)
    dtest = xgb.DMatrix(X_test, nthread=4)
    
    for i, model_path in enumerate(tqdm(fold_models, desc="Fold predictions")):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        bst = xgb.Booster()
        bst.load_model(str(model_path))
        
        # Predict
        pred = bst.predict(dtest)
        test_preds[:, i] = pred.astype(np.float32)
        
        del bst
        free_mem()
    
    # 4. Ensemble predictions (average)
    test_pred_avg = test_preds.mean(axis=1).astype(np.float32)
    print(f"\n[ENSEMBLE] Averaged predictions from {len(fold_models)} folds")
    print(f"  - Prediction range: [{test_pred_avg.min():.6f}, {test_pred_avg.max():.6f}]")
    print(f"  - Prediction mean: {test_pred_avg.mean():.6f}")
    
    # 5. Apply temperature calibration
    test_pred_calibrated = apply_temperature(test_pred_avg, temperature).astype(np.float32)
    print(f"\n[CALIBRATION] Applied temperature scaling (T={temperature:.3f})")
    print(f"  - Calibrated range: [{test_pred_calibrated.min():.6f}, {test_pred_calibrated.max():.6f}]")
    print(f"  - Calibrated mean: {test_pred_calibrated.mean():.6f}")
    
    # 6. Create submission file
    pred_df = pd.DataFrame({
        "ID": ids_test,
        "clicked": test_pred_calibrated
    })
    
    # Load sample submission to ensure correct format
    if Path(SAMPLE_SUB_PATH).exists():
        sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
        
        if "ID" in sample_sub.columns:
            sample_sub["ID"] = sample_sub["ID"].astype(str)
            submission = sample_sub[["ID"]].merge(pred_df, on="ID", how="left")
            
            # Check for missing predictions
            missing_mask = submission["clicked"].isna()
            if missing_mask.any():
                print(f"[WARNING] {missing_mask.sum()} rows have missing predictions, filling with mean")
                submission.loc[missing_mask, "clicked"] = test_pred_calibrated.mean()
        else:
            # If sample submission has no ID column, just use predictions in order
            submission = pd.DataFrame({"clicked": pred_df["clicked"].values})
    else:
        print(f"[WARNING] Sample submission not found at {SAMPLE_SUB_PATH}")
        print("[INFO] Creating submission with ID and clicked columns")
        submission = pred_df
    
    # Save submission
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    score_str = f"{metadata['oof_metrics']['score']:.5f}".replace(".", "p")
    output_filename = f"submission_{metadata['run_version']}_seed{metadata['seed']}_{score_str}_{timestamp}.csv"
    output_path = OUTPUT_DIR / output_filename
    
    submission.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"[SUCCESS] Submission file created!")
    print(f"  - Path: {output_path}")
    print(f"  - Shape: {submission.shape}")
    print(f"  - Columns: {list(submission.columns)}")
    print(f"{'='*60}")
    
    # Save prediction details for analysis
    detail_path = OUTPUT_DIR / f"prediction_details_{timestamp}.json"
    details = {
        "model_dir": str(MODEL_DIR),
        "run_version": metadata['run_version'],
        "seed": metadata['seed'],
        "temperature": temperature,
        "n_folds": len(fold_models),
        "oof_score": metadata['oof_metrics']['score'],
        "prediction_stats": {
            "min": float(test_pred_calibrated.min()),
            "max": float(test_pred_calibrated.max()),
            "mean": float(test_pred_calibrated.mean()),
            "median": float(np.median(test_pred_calibrated)),
            "std": float(test_pred_calibrated.std())
        },
        "output_file": str(output_path),
        "timestamp": timestamp
    }
    
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)
    
    print(f"\n[SAVE] Prediction details saved: {detail_path}")
    print("\n✓ Inference completed successfully!")

if __name__ == "__main__":
    main()