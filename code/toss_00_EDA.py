# -*- coding: utf-8 -*-

import polars as pl
import pandas as pd  # 시각화를 위해 최종 결과 변환용으로만 사용
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc

DATA_PATH = '/home/jjh/Project/competition/13_toss/data/'

def main():
    """
    Toss NEXT ML Challenge 데이터 분석 및 시각화 자동 저장 스크립트 (Polars 버전)
    """
    # 0. 준비 단계
    print("스크립트를 시작합니다.")
    output_dir = 'eda_plots_polars'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"'{output_dir}' 폴더에 분석 결과를 저장합니다.")

    # 데이터 로드 (Polars의 LazyFrame으로 메모리 사용 없이 스캔)
    try:
        lazy_df = pl.scan_parquet(os.path.join(DATA_PATH, 'train.parquet'))
        print("train.parquet 파일을 성공적으로 스캔했습니다. (Lazy Loading)")
    except FileNotFoundError:
        print(f"오류: '{os.path.join(DATA_PATH, 'train.parquet')}' 파일을 찾을 수 없습니다.")
        return

    # Matplotlib의 폰트 설정 (필요시 주석 해제)
    # plt.rcParams['font.family'] = 'Malgun Gothic'
    # plt.rcParams['axes.unicode_minus'] = False

    # 1. 타겟 변수 (`clicked`) 분석
    print("\n[1/6] 타겟 변수 분석 중...")
    target_dist = lazy_df.group_by('clicked').agg(pl.len().alias('count')).collect().to_pandas()
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x='clicked', y='count', data=target_dist)
    plt.title('Distribution of Target Variable (clicked)', fontsize=15)
    plt.xlabel('Clicked', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    total_count = target_dist['count'].sum()
    # 데이터에 0 또는 1만 있는 경우를 대비한 예외 처리
    if 0 in target_dist['clicked'].values:
        ratio_0 = target_dist[target_dist['clicked'] == 0]['count'].iloc[0] / total_count
        plt.text(0, target_dist[target_dist['clicked'] == 0]['count'].iloc[0]/2, f"0: {ratio_0:.2%}", ha='center')
    if 1 in target_dist['clicked'].values:
        ratio_1 = target_dist[target_dist['clicked'] == 1]['count'].iloc[0] / total_count
        plt.text(1, target_dist[target_dist['clicked'] == 1]['count'].iloc[0]/2, f"1: {ratio_1:.2%}", ha='center')
    plt.savefig(os.path.join(output_dir, '1_target_distribution.png'))
    plt.close()
    del target_dist
    gc.collect()

    # 2. 인구통계 및 주요 ID 피처 분석
    print("[2/6] 인구통계 및 주요 ID 피처 분석 중...")
    # 성별 분석
    gender_dist = lazy_df.group_by('gender').agg(pl.len().alias('count')).sort('gender').collect().to_pandas()
    gender_ctr = lazy_df.group_by('gender').agg(pl.col('clicked').mean()).sort('gender').collect().to_pandas()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.barplot(x='gender', y='count', data=gender_dist, ax=axes[0]).set_title('Gender Distribution')
    sns.barplot(x='gender', y='clicked', data=gender_ctr, ax=axes[1]).set_title('CTR by Gender')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_gender_analysis.png'))
    plt.close()

    # 연령 그룹 분석
    age_dist = lazy_df.group_by('age_group').agg(pl.len().alias('count')).sort('age_group').collect().to_pandas()
    age_ctr = lazy_df.group_by('age_group').agg(pl.col('clicked').mean()).sort('age_group').collect().to_pandas()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.barplot(x='age_group', y='count', data=age_dist, ax=axes[0]).set_title('Age Group Distribution')
    sns.barplot(x='age_group', y='clicked', data=age_ctr, ax=axes[1]).set_title('CTR by Age Group')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3_age_group_analysis.png'))
    plt.close()

    # 지면(inventory_id) 분석
    top_n = 20
    top_inventories = lazy_df.group_by('inventory_id').agg(pl.len().alias('count')).sort('count', descending=True).head(top_n).collect()
    # DeprecationWarning 방지를 위해 .to_list() 추가
    top_inventory_ids = top_inventories.get_column('inventory_id').to_list()
    inventory_ctr_pd = lazy_df.filter(pl.col('inventory_id').is_in(top_inventory_ids)).group_by('inventory_id').agg(
        pl.col('clicked').mean()
    ).sort('clicked', descending=True).collect().to_pandas()
    
    plt.figure(figsize=(14, 7))
    sns.barplot(x='inventory_id', y='clicked', data=inventory_ctr_pd)
    plt.title(f'CTR for Top {top_n} Inventories', fontsize=15)
    plt.xlabel('Inventory ID', fontsize=12)
    plt.ylabel('Click-Through Rate', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '4_top_inventories_ctr.png'))
    plt.close()
    del gender_dist, gender_ctr, age_dist, age_ctr, top_inventories, top_inventory_ids, inventory_ctr_pd
    gc.collect()

    # 3. 시간/순서 피처 분석
    print("[3/6] 시간/순서 피처 분석 중...")
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    # 숫자 문자열 -> 요일 이름 매핑 딕셔너리 생성 (1:월요일 ~ 7:일요일 가정)
    day_mapping = {str(i + 1): day for i, day in enumerate(days_order)}

    # Polars에서 순서가 있는 카테고리 타입으로 변환 (Enum 사용)
    lazy_df_ordered = lazy_df.with_columns(
        pl.col("day_of_week").replace(day_mapping).cast(pl.Enum(categories=days_order))
    )
    
    day_ctr_pd = lazy_df_ordered.group_by('day_of_week').agg(pl.col('clicked').mean()).sort('day_of_week').collect().to_pandas()
    hour_ctr_pd = lazy_df.group_by('hour').agg(pl.col('clicked').mean()).sort('hour').collect().to_pandas()

    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    sns.barplot(x='day_of_week', y='clicked', data=day_ctr_pd, ax=axes[0]).set_title('CTR by Day of Week', fontsize=15)
    sns.lineplot(x='hour', y='clicked', data=hour_ctr_pd, ax=axes[1], marker='o').set_title('CTR by Hour', fontsize=15)
    axes[1].grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '5_time_features_ctr.png'))
    plt.close()
    del day_ctr_pd, hour_ctr_pd, lazy_df_ordered
    gc.collect()

    # 4. 익명화된 피처(Anonymized Features) 분석
    print("[4/6] 익명화된 피처 분석 중...")
    anon_feats = [col for col in lazy_df.columns if 'feat_' in col and col != 'l_feat_14']
    cardinality_exprs = [pl.col(c).n_unique().alias(c) for c in anon_feats]
    feat_cardinality = lazy_df.select(cardinality_exprs).collect().transpose(include_header=True, column_names=['count']).sort('count', descending=True)
    
    with open(os.path.join(output_dir, '6_anonymized_feature_cardinality.txt'), 'w') as f:
        f.write("Cardinality of Anonymized Features:\n")
        f.write(str(feat_cardinality))

    history_feats = [col for col in lazy_df.columns if 'history_a_' in col]
    sample_hist_feats = history_feats[:2]
    fig, axes = plt.subplots(len(sample_hist_feats), 2, figsize=(15, 5 * len(sample_hist_feats)))
    for i, feat in enumerate(sample_hist_feats):
        # 각 컬럼을 필요한 만큼만 collect하여 메모리 사용 최소화
        hist_data = lazy_df.select([
            pl.col(feat), 
            pl.col(feat).log1p().alias(f"log_{feat}")
        ]).collect().to_pandas()
        
        sns.histplot(hist_data[feat], bins=50, kde=False, ax=axes[i, 0])
        axes[i, 0].set_title(f'Distribution of {feat}')
        sns.histplot(hist_data[f"log_{feat}"], bins=50, kde=False, ax=axes[i, 1])
        axes[i, 1].set_title(f'Log-Transformed Distribution of {feat}')
        del hist_data # 루프마다 메모리 정리
        gc.collect()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '7_history_feature_distribution.png'))
    plt.close()
    del anon_feats, feat_cardinality, history_feats
    gc.collect()

    # 5. 피처 간 관계 분석
    print("[5/6] 피처 간 관계 분석 중...")
    age_hour_ctr = lazy_df.group_by(['age_group', 'hour']).agg(pl.col('clicked').mean()).collect()
    age_hour_pivot = age_hour_ctr.pivot(index='age_group', columns='hour', values='clicked').sort('age_group')
    age_hour_pivot_pd = age_hour_pivot.to_pandas().set_index('age_group')
    
    plt.figure(figsize=(18, 9))
    sns.heatmap(age_hour_pivot_pd, cmap='viridis', annot=False)
    plt.title('CTR Heatmap by Age Group and Hour', fontsize=15)
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Age Group', fontsize=12)
    plt.savefig(os.path.join(output_dir, '8_heatmap_age_hour_ctr.png'))
    plt.close()
    del age_hour_ctr, age_hour_pivot, age_hour_pivot_pd
    gc.collect()

    # 6. 피처 엔지니어링 전략 시뮬레이션
    print("[6/6] 피처 엔지니어링 전략 시뮬레이션 및 코드 예시 생성 중...")
    fe_strategy = """
# Polars를 사용한 피처 엔지니어링 전략 코드 예시

# 1. 상호작용(Interaction) 피처
# lazy_df = lazy_df.with_columns([
#     (pl.col('gender').cast(str) + "_" + pl.col('age_group').cast(str)).alias('user_context')
# ])
# print("상호작용 피처 예시 생성 완료")

# 2. 통계(Aggregation) 피처
# user_id_feature = 'l_feat_1'
# user_agg = lazy_df.group_by(user_id_feature).agg([
#     pl.col('clicked').mean().alias('user_ctr'),
#     pl.len().alias('user_exposure_count') # pl.len() 사용
# ])
# lazy_df = lazy_df.join(user_agg, on=user_id_feature, how='left')
# print("통계 피처 예시 생성 완료")

# 3. 시간 기반 피처
# lazy_df = lazy_df.with_columns([
#     pl.col('day_of_week').is_in(['Saturday', 'Sunday']).cast(pl.Int8).alias('is_weekend'),
#     pl.when(pl.col('hour').is_between(6, 11)).then(pl.lit('Morning'))
#       .when(pl.col('hour').is_between(12, 17)).then(pl.lit('Afternoon'))
#       .when(pl.col('hour').is_between(18, 22)).then(pl.lit('Evening'))
#       .otherwise(pl.lit('Night')).alias('time_of_day')
# ])
# print("시간 기반 피처 예시 생성 완료")
    """
    with open(os.path.join(output_dir, '9_feature_engineering_examples_polars.txt'), 'w', encoding='utf-8') as f:
        f.write(fe_strategy)

    print("\n모든 분석 및 시각화가 완료되었습니다.")
    print(f"결과는 '{output_dir}' 폴더에서 확인하실 수 있습니다.")


if __name__ == '__main__':
    main()

