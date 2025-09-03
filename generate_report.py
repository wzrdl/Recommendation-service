# generate_report.py (v2 - Updated for latest Evidently AI)
import pandas as pd
from sklearn.model_selection import train_test_split

from evidently import Report
# 注意：这里的 'metric_presets' 是复数形式，这是新版本的API
from evidently.presets import DataDriftPreset

print("加载数据...")
# --- 数据加载与特征工程 (与train_model.py中完全相同) ---
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_df = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols)
movie_stats = ratings_df.groupby('movie_id')['rating'].agg(['mean', 'count'])
movie_stats.columns = ['movie_avg_rating', 'movie_rating_count']
user_stats = ratings_df.groupby('user_id')['rating'].agg(['mean', 'count'])
user_stats.columns = ['user_avg_rating', 'user_rating_count']
df = pd.merge(ratings_df, movie_stats, on='movie_id')
df = pd.merge(df, user_stats, on='user_id')
features = ['user_avg_rating', 'user_rating_count', 'movie_avg_rating', 'movie_rating_count']
X = df[features]
    
# 将数据分割，一份作为“参考”，一份作为“当前”
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
print("数据准备完成。")

# --- 生成Evidently AI报告 ---
print("正在生成数据漂移报告...")
# 1) 构建 Report
report = Report(metrics=[DataDriftPreset()])

# 2) 运行评估（两份数据：current 和 reference）
# 你之前的命名是 current=X_test, reference=X_train，这样写也OK
eval_result = report.run(current_data=X_test, reference_data=X_train)

# 3) 保存为 HTML（对 eval_result 调用）
report_filename = "data_drift_report.html"
eval_result.save_html(report_filename)

print(f"报告已生成: {report_filename}")
