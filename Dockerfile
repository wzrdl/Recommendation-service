# 1. 选择一个官方的Python作为基础镜像
# 我们选择一个包含Python 3.12的轻量级slim版本
FROM python:3.12-slim

# 2. 设置工作目录
# 这意味着后续所有命令都会在这个容器内的 /app 文件夹下执行
WORKDIR /app

# 3. 复制依赖项文件
# 我们先把 requirements.txt 文件复制进去，以便高效地利用Docker的缓存机制
COPY requirements.txt .

# 4. 安装Python依赖项
# --no-cache-dir 选项可以减小最终镜像的大小
RUN pip install --no-cache-dir -r requirements.txt

# 5. 复制我们自己的应用代码和模型文件到容器中
COPY . .

# 6. 暴露端口
# 告诉Docker，我们的应用将在容器的8000端口上监听
EXPOSE 8000

# 7. 定义启动命令
# 这是当容器启动时，会自动执行的命令
# 它会启动uvicorn服务器，运行我们的api应用
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
