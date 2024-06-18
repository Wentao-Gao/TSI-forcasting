# TSI: A multi-view representation learning approach for time series forecasting

## dataset 

dataset can be downloaded here: https://anonymous.4open.science/r/datasets-0541

## Evaluation Methodology
Consistent with established practices, our experiments focus on addressing multivariate forecasting scenarios. The multivariate setting involves analyzing data across multiple dimensionsWe employ Mean Squared Error (MSE) and Mean Absolute Error (MAE) as our evaluation metrics. The dataset is divided into training, validation, and testing segments in a 60/20/20 ratio. Following normalization to a zero mean, the performance is evaluated across different forecast lengths. Echoing the approach of \cite{yue2022ts2vec}, we initially train self-supervised learning models on the training set. Subsequently, a ridge regression model, built upon the acquired representations, is deployed for direct forecasting over the entire prediction length. The validation set is used to fine-tune the ridge regression regularization parameter \( \alpha \), exploring values within \{0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000\}. The test set is employed for final result analysis.


## Loss function

***For trend Representation***

For each sample \( G_{j} \) at a given time step \( t \), we first randomly select a time step for the contrastive loss and then apply a projection head—specifically, a single-layer Multilayer Perceptron (MLP)—to obtain an augmented query vector \( q \). Additionally, the positive key \( k^+ \) is derived from the same time step \( t \) using a momentum encoder, to ensure temporal consistency. In contrast, the negative keys \( k^- \) are obtained by a dynamic dictionary. This allows the model to identify not only samples that are temporally close to the time step \( t \) (i.e., the positive samples) but also to discriminate against those that are temporally distant (i.e., the negative samples). We compute the time-domain contrastive loss using the following formula:

<img width="410" alt="截屏2024-06-18 下午10 29 00" src="https://github.com/Wentao-Gao/TSI-forcasting/assets/77952995/e7bb7898-7b95-4049-8fbc-ade793f6380e">

In this equation, \( q_i \) represents the query vector, \( k^+_i \) is the key vector of the positive sample that is temporally close to \( q_i \) at time \( t \), \( k^-_{n} \) are the key vectors of the negative samples from other time steps, \( N' \) is the number of negative samples, and \( \tau \) is a temperature parameter that modulates the distinction between positive and negative samples.

***For Seasonal representation***


Training is supplemented by a novel contrastive approach inspired by \cite{woo2022cost} in the frequency domain, which we term the Frequency Domain Contrastive Loss. This loss function is vital for contrasting the amplitude and phase components within the frequency spectrum of the data. The contrastive loss can be divided into two distinct components: amplitude contrast \( \mathcal{L}_{\text{amp}} \) and phase contrast \( \mathcal{L}_{\text{phase}} \), which are defined as follows:


<img width="360" alt="截屏2024-06-18 下午10 23 06" src="https://github.com/Wentao-Gao/TSI-forcasting/assets/77952995/13cf196a-f7d2-4ada-a08a-467147c1f908">


In these equations, \( F_{i,:}^{(n)} \) refers to the \( j \)-th sample's frequency representation in a mini-batch, and \( F_{i,:}^{(n)'} \) is its augmented counterpart, thereby forming the basis for the contrastive comparison. 

## Data augmentation strategies

**Scaling**
To scale the time series, we draw a random scalar from a normal distribution $\epsilon \sim \mathcal{N}(0, 0.5^2)$. This scalar is then used to uniformly scale every data point in the series, thus the transformed time step becomes $\tilde{x}_t = \epsilon x_t$.

**Shifting**
Shifting involves adding a random scalar to each time step, simulating a vertical translation of the entire series. This scalar is also sampled from a normal distribution $\epsilon \sim \mathcal{N}(0, 0.5^2)$, leading to the adjusted time step $\tilde{x}_t = x_t + \epsilon$.

**Jittering**
Lastly, jittering introduces variability by injecting independent and identically distributed (I.I.D.) Gaussian noise into each time step. For every time step, noise $\epsilon_t \sim \mathcal{N}(0, 0.5^2)$ is added, resulting in the final time step $\tilde{x}_t = x_t + \epsilon_t$.


## Ablation study

1. For components

<img width="559" alt="截屏2024-06-18 下午10 15 34" src="https://github.com/Wentao-Gao/TSI-forcasting/assets/77952995/6fe2a438-1966-441e-be52-2e5f4c09f4ab">


In our ablation study, we dissected the impact of different components within our time series forecasting model, namely the Independent, Trend, and Seasonality components, alongside their combined form referred to as TSI (Trend + Seasonality + Independent). The evaluation focused on Mean Squared Error (MSE) and Mean Absolute Error (MAE) metrics for multivariate tasks, with the results detailed in table \ref{tab:component}.

Within the multivariate framework, the TSI component markedly surpassed the baseline Independent component, evidenced by an improvement in MSE from 1.000 to 0.622 and in MAE from 0.748 to 0.571. This significant enhancement underscores the utility of integrating various time series components to adeptly capture the complexities inherent in multivariate data dynamics. The ablation study's findings advocate for a holistic approach in time series analysis, particularly highlighting the superior performance achieved through the TSI component in multivariate settings.

2. For predictors
3. 
<img width="563" alt="截屏2024-06-18 下午10 15 54" src="https://github.com/Wentao-Gao/TSI-forcasting/assets/77952995/9f5e6279-a3b7-4c86-9be5-bd2377190c64">

In this ablation study, our objective was to evaluate the effectiveness of various regression models in capturing essential information within multivariate time series data. Through a comprehensive comparison of different models, we highlighted the superior capability of our model in extracting critical insights, as demonstrated by its outstanding performance against traditional regression techniques.


The results, as depicted in table \ref{tab:predictor}, illustrate that our model consistently outperforms standard regression methods in multivariate forecasting scenarios. In particular, our approach achieved a notable Mean Squared Error (MSE) of 0.622 and a Mean Absolute Error (MAE) of 0.571. This indicates our model's advanced ability to identify and leverage underlying patterns within complex datasets, affirming its proficiency in multivariate analysis.


The ablation study underscores the distinct advantages of our model, especially when integrated with ridge regression, highlighting its effectiveness in extracting vital information from the data. This demonstrates our model's significance as a powerful tool for multivariate time series forecasting, offering substantial enhancements over conventional methods and setting a new benchmark in the field.

