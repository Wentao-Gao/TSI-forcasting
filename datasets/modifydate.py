import pandas as pd

# 读取CSV文件
df = pd.read_csv('weather_case.csv')

# 删除'lat'和'lon'列
df = df.drop(columns=['lat', 'lon'])

# 将'time'列的名称更改为'date'
df = df.rename(columns={'time': 'date'})

# 保存修改后的DataFrame回CSV文件
df.to_csv('weather_case.csv', index=False)

