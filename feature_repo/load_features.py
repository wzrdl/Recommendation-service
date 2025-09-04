# load_features.py
import pandas as pd
from feast import FeatureStore
from datetime import datetime, timezone

def load_features():
    """Load and write features to the online store"""
    print("正在加载特征仓库...")
    # 加载当前目录下的特征仓库
    store = FeatureStore(repo_path=".")

    print("正在计算聚合特征...")
    # --- 这部分计算逻辑与 train_model.py 完全相同 ---
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings_df = pd.read_csv('../ml-100k/u.data', sep='\t', names=r_cols)
    movie_stats = ratings_df.groupby('movie_id')['rating'].agg(['mean', 'count']).reset_index()
    movie_stats.columns = ['movie_id', 'movie_avg_rating', 'movie_rating_count']
    user_stats = ratings_df.groupby('user_id')['rating'].agg(['mean', 'count']).reset_index()
    user_stats.columns = ['user_id', 'user_avg_rating', 'user_rating_count']

    # --- 为特征数据加入事件时间戳 ---
    # Feast在写入线上商店时，需要知道每个特征值是何时生成的
    now = datetime.now(timezone.utc)
    user_stats["event_timestamp"] = now
    movie_stats["event_timestamp"] = now

    print("正在将特征写入线上商店 (Redis)...")
    # --- 使用 Feast 的 write_to_online_store 方法写入 ---
    store.write_to_online_store(
        feature_view_name="user_features",
        df=user_stats,
    )
    store.write_to_online_store(
        feature_view_name="movie_features",
        df=movie_stats,
    )

    print("特征加载完成。")

if __name__ == "__main__":
    load_features()