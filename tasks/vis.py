import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from models.iencoder import load_and_preprocess_ica_data, NonlinearICA, train_ica_model


file_path = r'C:\Users\gwt99\PycharmProjects\TSI(Trend Seasonality Indenpendent)\datasets\ETTh1.csv'
train_X, valid_X, test_X = load_and_preprocess_ica_data(file_path)

input_dim = train_X.shape[1]
hidden_dim = 100
    #source_dim = train_X.shape[1]
source_dim = 100

ica_model = NonlinearICA(input_dim, hidden_dim, source_dim)
train_ica_model(ica_model, train_X)

train_source_data = ica_model.encode(train_X.view(train_X.shape[0], -1)).detach().numpy()
valid_source_data = ica_model.encode(valid_X.view(valid_X.shape[0], -1)).detach().numpy()
test_source_data = ica_model.encode(test_X.view(test_X.shape[0], -1)).detach().numpy()


# 假设 X 是您的高维特征矩阵，每行是一个数据点的表征
X = test_source_data   # 替换为您的表征数据

# 数据预处理：标准化特征
X_scaled = StandardScaler().fit_transform(X)

# 应用 t-SNE
tsne = TSNE(n_components=2, random_state=0)
X_tsne = tsne.fit_transform(X_scaled)

# 可视化
plt.figure(figsize=(12, 8))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], marker='.', alpha=0.7)
plt.title('t-SNE visualization of representations')
plt.xlabel('t-SNE feature 1')
plt.ylabel('t-SNE feature 2')
plt.show()
