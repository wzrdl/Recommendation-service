import pandas as pd

# 定义列名
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
# 加载评分数据 u.data
ratings_df = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols)

# 将评分 (1-5) 转化为我们需要的二元标签 (0或1)
# 我们定义：评分>=4为喜欢(1), 否则为不喜欢(0)
ratings_df['like'] = (ratings_df['rating'] >= 4).astype(int)

print("处理后的评分数据示例:")
print(ratings_df.head())

# 定义电影元数据文件的列名
i_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure',
          'Animation', 'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
          'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

# 加载电影元数据 u.item
movies_df = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols, encoding='latin-1')

# 将电影数据和评分数据合并在一起
# 这就像数据库中的 JOIN 操作
movie_ratings_df = pd.merge(ratings_df, movies_df, on='movie_id')

print("合并后的数据表示例:")
print(movie_ratings_df.head())
