# api.py
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# 1. 初始化 FastAPI 应用
app = FastAPI(title="Movie Recommendation API", version="1.0")

# 2. 定义输入数据的模型 (用于数据验证)
#    这确保了发往API的请求必须包含这四个字段，且它们都是浮点数。
class Features(BaseModel):
    user_avg_rating: float
    user_rating_count: float
    movie_avg_rating: float
    movie_rating_count: float

# 3. 加载训练好的模型
#    模型在API启动时只加载一次，以提高效率。
model_filename = 'xgb_model.joblib'
try:
    model = joblib.load(model_filename)
    print("模型加载成功。")
except FileNotFoundError:
    print(f"错误: 模型文件 '{model_filename}' 未找到。请先运行 train_model.py。")
    model = None

# 4. 定义预测端点
#    @app.post("/predict") 表示这是一个接受POST请求的端点。
@app.post("/predict")
def predict(features: Features):
    if not model:
        return {"error": "模型未能加载，无法进行预测。"}

    # 将输入的特征转化为DataFrame，因为模型期望的是DataFrame格式
    input_df = pd.DataFrame([features.dict()])
    
    # 使用模型进行预测，predict_proba会返回[不喜欢概率, 喜欢概率]
    prediction_proba = model.predict_proba(input_df)[0]
    
    # 我们只关心“喜欢”的概率
    like_probability = prediction_proba[1]
    
    # 以JSON格式返回结果
    return {"like_probability": float(like_probability)}

# 5. 定义一个根端点 (可选，用于测试服务是否在线)
@app.get("/")
def read_root():
    return {"status": "Movie Recommendation API is online."}
