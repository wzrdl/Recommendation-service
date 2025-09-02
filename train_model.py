import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

# --- 1. 数据加载和预处理 ---
# 定义列名
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
# 加载评分数据 u.data
ratings_df = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols)

# 将评分 (1-5) 转化为二元标签 (0或1)
ratings_df['like'] = (ratings_df['rating'] >= 4).astype(int)

# --- 2. 特征工程 ---
print("开始进行特征工程...")
# 计算电影的统计特征
movie_stats = ratings_df.groupby('movie_id')['rating'].agg(['mean', 'count'])
movie_stats.columns = ['movie_avg_rating', 'movie_rating_count']

# 计算用户的统计特征
user_stats = ratings_df.groupby('user_id')['rating'].agg(['mean', 'count'])
user_stats.columns = ['user_avg_rating', 'user_rating_count']

# 将特征合并回主数据框
df = pd.merge(ratings_df, movie_stats, on='movie_id')
df = pd.merge(df, user_stats, on='user_id')

print("特征工程完成。")

# --- 3. 准备训练数据 ---
# 定义模型的特征(X)和目标(y)
features = ['user_avg_rating', 'user_rating_count', 'movie_avg_rating', 'movie_rating_count']
target = 'like'

X = df[features]
y = df[target]

# 将数据分割为训练集 (80%) 和测试集 (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"数据准备完成。训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")

# --- 4. 模型训练 ---
print("开始训练XGBoost模型...")
# 初始化XGBoost分类器
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# 训练模型
model.fit(X_train, y_train)

print("模型训练完成。")

# --- 5. 模型评估 ---
print("开始评估模型...")
# 在测试集上进行预测
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# 计算并打印评估指标
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"模型评估完成。")
print(f"  - 准确率 (Accuracy): {accuracy:.4f}")
print(f"  - AUC (Area Under Curve): {auc:.4f}")

# --- 6. 模型序列化 (保存) ---
model_filename = 'xgb_model.joblib'
joblib.dump(model, model_filename)

print(f"模型已成功保存到文件: {model_filename}")
