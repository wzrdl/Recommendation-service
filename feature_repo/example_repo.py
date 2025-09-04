# example_repo.py (v3 - Final Aggregated Features)
from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Int64

# 数据源现在是ratings.parquet，因为它包含了我们需要的事件时间戳
# 但请注意：Feast不会用这个文件来计算平均值，我们稍后会手动加载这些特征
ratings_source = FileSource(
    path="../ml-100k/ratings.parquet",
    timestamp_field="event_timestamp",
)

# 定义实体
user = Entity(name="user_id", value_type=ValueType.INT64)
movie = Entity(name="movie_id", value_type=ValueType.INT64)

# 定义用户特征视图
# 这次我们定义了模型真正需要的聚合特征
user_features_fv = FeatureView(
    name="user_features",
    entities=[user],
    ttl=timedelta(days=365),
    schema=[
        Field(name="user_avg_rating", dtype=Float32),
        Field(name="user_rating_count", dtype=Int64),
    ],
    source=ratings_source,
)

# 定义电影特征视图
movie_features_fv = FeatureView(
    name="movie_features",
    entities=[movie],
    ttl=timedelta(days=365),
    schema=[
        Field(name="movie_avg_rating", dtype=Float32),
        Field(name="movie_rating_count", dtype=Int64),
    ],
    source=ratings_source,
)