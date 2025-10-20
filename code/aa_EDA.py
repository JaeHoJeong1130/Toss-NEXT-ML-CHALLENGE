import polars as pl
import pandas as pd

# Pandas/Polars 출력 옵션 설정
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pl.Config.set_tbl_rows(10) # polars 출력 행 수 조절
pl.Config.set_tbl_cols(20) # polars 출력 열 수 조절

dPATH = '/home/jjh/Project/competition/13_toss/data/'
TRAIN_PATH = dPATH + 'train.parquet'

print(f"'{TRAIN_PATH}' 파일 탐색을 시작합니다.")

# 1. scan_parquet으로 파일을 메모리에 올리지 않고 스캔
lazy_df = pl.scan_parquet(TRAIN_PATH)
schema = lazy_df.collect_schema()

print("\n--- 1. 데이터 Shape 및 기본 정보 ---")
print(f"행(Rows): {lazy_df.select(pl.len()).collect().item():,}")
print(f"열(Columns): {len(schema.names())}")

# 2. 데이터 미리보기 (가독성 개선)
print("\n--- 2. 데이터 미리보기 ---")

print("\n[방법 1] 첫 번째 행을 세로로 길게 보기 (모든 컬럼 확인용)")
# 첫 번째 행만 가져와서 전치(transpose)하여 출력
first_row = lazy_df.head(1).collect()
print(first_row.transpose(include_header=True, header_name="column", column_names=["value_of_first_row"]))

print("\n[방법 2] 컬럼 그룹별로 나누어 보기")
print("\n>>> 기본 정보 컬럼 (head 5)")
basic_cols = ['gender', 'age_group', 'inventory_id', 'day_of_week', 'hour', 'seq', 'clicked']
print(lazy_df.select(basic_cols).head(5).collect())

print("\n>>> 'l_feat_*' 컬럼 (head 5)")
# 정규표현식을 사용하여 'l_feat_'로 시작하는 모든 컬럼 선택
l_feat_cols = lazy_df.select(pl.col('^l_feat_.*$')).columns
print(lazy_df.select(l_feat_cols).head(5).collect())

print("\n>>> 'history_a_*' 컬럼 (head 5)")
history_a_cols = lazy_df.select(pl.col('^history_a_.*$')).columns
print(lazy_df.select(history_a_cols).head(5).collect())

# 3. 수치형 데이터 기본 통계량 확인
print("\n--- 3. 수치형(Numerical) 데이터 통계 (Describe) ---")
# pyarrow가 설치되면 이 부분은 정상적으로 작동합니다.
try:
    print(lazy_df.describe().to_pandas())
except ImportError:
    print("오류: 'pyarrow'가 설치되지 않았습니다. 'pip install pyarrow'를 실행해주세요.")

# 4. 컬럼별 결측치(Null) 개수 확인
print("\n--- 4. 컬럼별 결측치(Null) 개수 ---")
null_counts = lazy_df.null_count().collect()
non_zero_nulls = False
for col in null_counts.columns:
    count = null_counts[col][0]
    if count > 0:
        print(f"- {col}: {count:,} 개")
        non_zero_nulls = True
if not non_zero_nulls:
    print("모든 컬럼에 결측치가 없습니다.")