# train_model.py (v5 - 集成Optuna和MLflow)
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import joblib
import mlflow
import mlflow.xgboost
import os
import optuna  # 导入optuna

# --- MLflow 配置 ---
MLFLOW_SERVER_IP = os.getenv("MLFLOW_SERVER_IP", "3.18.105.42")
MLFLOW_TRACKING_URI = f"http://{MLFLOW_SERVER_IP}:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Movie Recommender - XGBoost Hyperparameter Tuning")

# --- 数据加载与特征工程 (这部分保持不变) ---
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_df = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols)
ratings_df['like'] = (ratings_df['rating'] >= 4).astype(int)
movie_stats = ratings_df.groupby('movie_id')['rating'].agg(['mean', 'count'])
movie_stats.columns = ['movie_avg_rating', 'movie_rating_count']
user_stats = ratings_df.groupby('user_id')['rating'].agg(['mean', 'count'])
user_stats.columns = ['user_avg_rating', 'user_rating_count']
df = pd.merge(ratings_df, movie_stats, on='movie_id')
df = pd.merge(df, user_stats, on='user_id')
features = ['user_avg_rating', 'user_rating_count', 'movie_avg_rating', 'movie_rating_count']
target = 'like'
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# --- 1. 定义Optuna的目标函数 (Objective Function) ---
def objective(trial):
    # Optuna的每一次尝试，都会被记录为一个MLflow的子运行
    with mlflow.start_run(nested=True):
        # 2. 定义超参数的搜索空间
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            # Optuna会从100到1000之间智能地选择一个整数
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            # Optuna会从0.01到0.3之间智能地选择一个浮点数
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            # Optuna会从3到10之间智能地选择一个整数
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'seed': 42
        }

        # 记录这次尝试的参数
        mlflow.log_params(params)

        # 训练和评估模型
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)

        # 记录这次尝试的性能
        mlflow.log_metric("auc", auc)

        # 3. 返回性能指标，Optuna会尝试最大化这个值
        return auc

# --- 4. 启动Optuna的优化过程 ---
if __name__ == "__main__":
    # 创建一个父级的MLflow Run，来管理整个优化过程
    with mlflow.start_run() as parent_run:
        print("Parent Run ID:", parent_run.info.run_id)
        # 创建一个Optuna study，目标是最大化 (maximize) objective函数的返回值
        study = optuna.create_study(direction='maximize')
        # 启动优化，n_trials=20 表示Optuna会进行20次尝试
        study.optimize(objective, n_trials=2)

        print("\n优化过程完成。")
        print("尝试次数: ", len(study.trials))
        print("最佳AUC值: ", study.best_value)
        print("最佳参数组合: ", study.best_params)
