# convert_to_parquet.py
import pandas as pd

print("正在读取TSV文件...")
# 定义列名并加载u.data文件
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_df = pd.read_csv('../ml-100k/u.data', sep='\t', names=r_cols)

# Feast需要一个datetime类型的timestamp列
ratings_df['event_timestamp'] = pd.to_datetime(ratings_df['unix_timestamp'], unit='s')

# Feast需要一个创建时间的列
ratings_df['created'] = pd.to_datetime('now', utc=True)

output_path = "../ml-100k/ratings.parquet"
print(f"正在将数据转换为Parquet格式，并保存到 {output_path} ...")

# 将DataFrame保存为Parquet格式
ratings_df.to_parquet(output_path)

print("转换完成。")
