# streaming_job.py (v2 - with feature calculation logic)
import json
import pandas as pd
from kafka import KafkaConsumer
from feast import FeatureStore
from datetime import datetime

KAFKA_TOPIC = "click-stream"
# 注意：因为这个脚本和Kafka容器都在同一个服务器上，
# 并且不在同一个docker-compose网络中，所以我们用localhost连接
KAFKA_SERVER = "localhost:9092" 

# 1. 在内存中初始化一个“状态存储”
#    - key: user_id
#    - value: 一个包含 'total_rating', 'rating_count' 的字典
user_state = {}

def update_features(event_data):
    """
    根据新的事件，实时计算或更新用户的聚合特征。
    """
    user_id = event_data['user_id']
    rating = event_data.get('rating') # 假设我们的事件现在也包含评分

    if rating is None:
        return None # 如果事件没有评分信息，则跳过

    # 如果是新用户，初始化状态
    if user_id not in user_state:
        user_state[user_id] = {'total_rating': 0, 'rating_count': 0}

    # 更新状态
    user_state[user_id]['total_rating'] += rating
    user_state[user_id]['rating_count'] += 1

    # 计算新的聚合特征
    new_avg_rating = user_state[user_id]['total_rating'] / user_state[user_id]['rating_count']
    new_rating_count = user_state[user_id]['rating_count']
    
    # 准备要写入Feast的数据
    feature_df = pd.DataFrame({
        "user_id": [user_id],
        "event_timestamp": [datetime.utcnow()],
        "user_avg_rating": [new_avg_rating],
        "user_rating_count": [new_rating_count]
    })
    
    return feature_df

def main():
    print("正在连接到 Kafka...")
    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_SERVER,
        auto_offset_reset='earliest',
        group_id='feature_ingestion_group',
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )
    
    print("正在加载 Feast 特征仓库...")
    store = FeatureStore(repo_path="./feature_repo")

    print("开始监听来自 Kafka 的实时事件...")
    for message in consumer:
        event_data = message.value
        print(f"\n接收到事件: {event_data}")
        
        # 2. 计算新特征
        features_to_write = update_features(event_data)
        
        # 3. 如果成功计算出新特征，则写入Feast
        if features_to_write is not None:
            try:
                store.write_to_online_store(
                    feature_view_name="user_features",
                    df=features_to_write,
                )
                print("成功将更新后的特征写入 Feast:")
                print(features_to_write)
            except Exception as e:
                print(f"写入 Feast 时发生错误: {e}")

if __name__ == "__main__":
    main()