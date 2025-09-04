# api.py (v2 - with Feast online feature fetching)
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from feast import FeatureStore # 导入Feast

# 初始化 FastAPI 应用
app = FastAPI(title="Movie Recommendation API - Online")

# 1. 定义新的输入数据模型
#    现在API只需要user_id和movie_id
class PredictionRequest(BaseModel):
    user_id: int
    movie_id: int

# 2. 加载特征仓库和模型
#    告诉Feast我们的特征仓库在哪个文件夹
store = FeatureStore(repo_path="./feature_repo")
model = joblib.load('./feature_repo/xgb_model.joblib') # 假设模型文件也在feature_repo下

# 3. 定义改造后的预测端点
@app.post("/predict")
def predict(request: PredictionRequest):
    # 4. 定义我们需要从Feast中获取哪些特征
    feature_vector = store.get_online_features(
        features=[
            "user_features:user_avg_rating",
            "user_features:user_rating_count",
            "movie_features:movie_avg_rating",
            "movie_features:movie_rating_count",
        ],
        entity_rows=[
            {"user_id": request.user_id, "movie_id": request.movie_id}
        ],
    ).to_dict()

    # 5. 将从Feast获取的特征转换为模型需要的格式
    #    注意：特征的顺序必须和训练时完全一致
    features_df = pd.DataFrame.from_dict(feature_vector)
    ordered_features = [
        'user_avg_rating', 
        'user_rating_count', 
        'movie_avg_rating', 
        'movie_rating_count'
    ]

    # 检查所有特征是否都成功获取
    if not all(feature in features_df.columns for feature in ordered_features):
        return {"error": "Could not retrieve all required features from the feature store."}

    prediction_input = features_df[ordered_features]

    # 6. 使用模型进行预测
    prediction_proba = model.predict_proba(prediction_input)[0]
    like_probability = prediction_proba[1]

    return {"like_probability": float(like_probability)}

@app.get("/")
def read_root():
    return {"status": "Movie Recommendation Online API is running."}