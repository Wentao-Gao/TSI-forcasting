import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, train_test_split


def fit_ridge(train_features, train_y, valid_features, valid_y, MAX_SAMPLES=100000):
    # If the training set is too large, subsample MAX_SAMPLES examples
    if train_features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            train_features, train_y,
            train_size=MAX_SAMPLES, random_state=0
        )
        train_features = split[0]
        train_y = split[2]
    if valid_features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            valid_features, valid_y,
            train_size=MAX_SAMPLES, random_state=0
        )
        valid_features = split[0]
        valid_y = split[2]
    alphas = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    valid_results = []
    for alpha in alphas:
        lr = Ridge(alpha=alpha).fit(train_features, train_y)
        valid_pred = lr.predict(valid_features)
        score = np.sqrt(((valid_pred - valid_y) ** 2).mean()) + np.abs(valid_pred - valid_y).mean()
        valid_results.append(score)
    best_alpha = alphas[np.argmin(valid_results)]
    
    lr = Ridge(alpha=best_alpha)
    lr.fit(train_features, train_y)
    return lr


from sklearn.linear_model import LinearRegression



def fit_linear(train_features, train_y, valid_features, valid_y, MAX_SAMPLES=100000):
    # 如果训练集太大，则采样最多 MAX_SAMPLES 个样本
    if train_features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            train_features, train_y,
            train_size=MAX_SAMPLES, random_state=0
        )
        train_features, _, train_y, _ = split

    if valid_features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            valid_features, valid_y,
            train_size=MAX_SAMPLES, random_state=0
        )
        valid_features, _, valid_y, _ = split

    # 训练线性回归模型
    lr = LinearRegression().fit(train_features, train_y)

    return lr

from sklearn.kernel_ridge import KernelRidge


def fit_kernel_ridge(train_features, train_y, valid_features, valid_y, MAX_SAMPLES=100000):
    # 如果训练集太大，就采样最多 MAX_SAMPLES 个样本
    if train_features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            train_features, train_y,
            train_size=MAX_SAMPLES, random_state=0
        )
        train_features, _, train_y, _ = split

    if valid_features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            valid_features, valid_y,
            train_size=MAX_SAMPLES, random_state=0
        )
        valid_features, _, valid_y, _ = split

    # 定义一组可能的 alpha（正则化参数）和 gamma（用于某些核函数的参数）
    alphas = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    gammas = [0.1, 0.01, 0.001, 0.0001]

    best_score = float("inf")
    best_alpha, best_gamma = None, None

    # 交叉验证，找到最佳的 alpha 和 gamma
    for alpha in alphas:
        for gamma in gammas:
            kr = KernelRidge(alpha=alpha, kernel='rbf', gamma=gamma)
            kr.fit(train_features, train_y)
            valid_pred = kr.predict(valid_features)
            score = np.sqrt(((valid_pred - valid_y) ** 2).mean()) + np.abs(valid_pred - valid_y).mean()

            if score < best_score:
                best_score = score
                best_alpha, best_gamma = alpha, gamma

    # 使用最佳的 alpha 和 gamma 重新训练模型
    kr = KernelRidge(alpha=best_alpha, kernel='rbf', gamma=best_gamma)
    kr.fit(train_features, train_y)

    return kr
