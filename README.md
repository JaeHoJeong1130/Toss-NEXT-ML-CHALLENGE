# Toss-NEXT-ML-CHALLENGE

## 프로젝트 개요

토스 NEXT ML CHALLENGE: 광고 클릭 예측(CTR) 모델 개발

광고 데이터를 분석하여 클릭률(CTR)을 예측하는 머신러닝 프로젝트입니다.

## 주요 기능

- EDA (탐색적 데이터 분석)
- 피처 엔지니어링
- Target Encoding with K-Fold
- 시퀀스 피처 강화 (Polars List 연산)
- 다중 모델 앙상블
- 리포트 생성 및 시각화

## 파일 구조

```
Toss-NEXT-ML-CHALLENGE/
├── code/
│   ├── toss_00.py ~ toss_04.py
│   ├── aa_*.py              # 전처리 및 학습 코드
│   ├── 00_report_figures.py # 리포트 생성
│   └── ...
├── [대회]정재호_Toss_보고서.pdf
└── README.md
```

## 기술 스택

- Python
- XGBoost, LightGBM, CatBoost
- Polars
- scikit-learn
- matplotlib (시각화)