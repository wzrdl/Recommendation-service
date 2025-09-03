# example_repo.py (v2 - 使用Parquet)
from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Int64

# 1. 定义数据源，直接指向新的Parquet文件
# 我们不再需要指定文件格式，因为Parquet是默认的
ratings_source = FileSource(
    path="../ml-100k/ratings.parquet", # 指向新的Parquet文件
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)

# 2. 定义实体 (保持不变)
user = Entity(name="user_id", value_type=ValueType.INT64)
movie = Entity(name="movie_id", value_type=ValueType.INT64)

# 3. 定义特征视图 (这里的schema需要和Parquet文件中的列名完全对应)
# 我们的Parquet文件里没有user_avg_rating, movie_avg_rating这些列
# 我们先定义一个只包含原始评分的视图
ratings_fv = FeatureView(
    name="ratings",
    entities=[user, movie], # 这个视图同时关联用户和电影
    ttl=timedelta(days=365),
    schema=[
        Field(name="rating", dtype=Int64),
    ],
    source=ratings_source,
)
