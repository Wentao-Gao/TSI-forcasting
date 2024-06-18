# This is for the ECAI paper

## dataset 

dataset can be downloaded here: https://anonymous.4open.science/r/datasets-0541

## Evaluation Methodology
Consistent with established practices, our experiments focus on addressing multivariate forecasting scenarios. The multivariate setting involves analyzing data across multiple dimensionsWe employ Mean Squared Error (MSE) and Mean Absolute Error (MAE) as our evaluation metrics. The dataset is divided into training, validation, and testing segments in a 60/20/20 ratio. Following normalization to a zero mean, the performance is evaluated across different forecast lengths. Echoing the approach of \cite{yue2022ts2vec}, we initially train self-supervised learning models on the training set. Subsequently, a ridge regression model, built upon the acquired representations, is deployed for direct forecasting over the entire prediction length. The validation set is used to fine-tune the ridge regression regularization parameter \( \alpha \), exploring values within \{0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000\}. The test set is employed for final result analysis.
