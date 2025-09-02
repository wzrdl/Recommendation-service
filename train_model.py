# train_model.py (v4 - 使用 log_artifact)
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import joblib
import mlflow
import os

# --- 0. MLflow 配置 ---
# 从环境变量读取MLFLOW服务器地址，如果不存在，则使用您的IP作为默认值
MLFLOW_SERVER_IP = os.getenv("MLFLOW_SERVER_IP")
MLFLOW_TRACKING_URI = f"http://{MLFLOW_SERVER_IP}:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Movie Recommender - XGBoost")

# --- 1. 数据加载和预处理 ---
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_df = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols)
ratings_df['like'] = (ratings_df['rating'] >= 4).astype(int)

# --- 2. 特征工程 ---
movie_stats = ratings_df.groupby('movie_id')['rating'].agg(['mean', 'count'])
movie_stats.columns = ['movie_avg_rating', 'movie_rating_count']
user_stats = ratings_df.groupby('user_id')['rating'].agg(['mean', 'count'])
user_stats.columns = ['user_avg_rating', 'user_rating_count']
df = pd.merge(ratings_df, movie_stats, on='movie_id')
df = pd.merge(df, user_stats, on='user_id')

# --- 3. 准备训练数据 ---
features = ['user_avg_rating', 'user_rating_count', 'movie_avg_rating', 'movie_rating_count']
target = 'like'
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. 启动一个MLflow运行 (Run) ---
with mlflow.start_run() as run:
    print("MLflow Run ID:", run.info.run_id)

    # --- 4.1 模型训练 ---
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'n_estimators': 200,
        'learning_rate': 0.1,
        'max_depth': 4,
        'seed': 42
    }
    
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)

    # --- 4.2 模型评估 ---
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)

    # --- 4.3 在MLflow中记录参数和指标 ---
    print(f"AUC: {auc:.4f}")
    mlflow.log_params(params)
    mlflow.log_metric("auc", auc)
    
    # =================================================================
    # vvv 这里是新的模型记录方法 vvv
    # 4.4 先将模型保存到本地文件
    model_filename = "xgb_model.joblib"
    joblib.dump(model, model_filename)
    print(f"模型已成功保存到本地文件: {model_filename}")

    # 4.5 使用 log_artifact 手动上传模型文件
    # 第一个参数是本地文件路径，第二个参数是在MLflow UI中显示的文件夹名
    mlflow.log_artifact(model_filename, "model")
    print("模型文件已成功作为 artifact 记录到 MLflow。")
    # ^^^ 到这里结束 ^^^
    # =================================================================

print("MLflow run completed.")
