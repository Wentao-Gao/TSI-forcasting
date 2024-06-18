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



## Details on Baselines

Results on TS2Vec, TNC, CoST, Informer, LogTrans, and TCN are based on our reproduction.

- **TS2Vec**: TS2Vec, a recent innovation, presents a universal framework for learning time series representations. It utilizes contrastive learning in a hierarchical structure across augmented context views, focusing on learning representations at the timestamp level. We utilized the framework as provided in their open-source repository, adhering to the default hyper-parameters recommended in their paper. GitHub: [TS2Vec](https://github.com/yuezhihan/ts2vec)

- **TNC**: Temporal Neighborhood Coding (TNC) is a self-supervised framework that emphasizes the temporal neighborhood structure within time series. It effectively captures temporal dynamics by analyzing patterns of similarity and dissimilarity. We use the encoder of TS2Vec rather than their original encoders (CNN and RNN) as its backbone. Previous work has proven that this would have better performance than the original, as this can be attributed to the adaptive receptive fields of dilated convolutions, which better fit datasets from various scales. GitHub: [TNC](https://github.com/sanatonek/TNC_representation_learning)

- **CoST**: CoST employs a transformer-based architecture, combined with the principles of contrastive learning, to effectively process time series data. This method is particularly adept at identifying and understanding long-range dependencies within the data, a crucial aspect in time series analysis. CoST's design allows for a nuanced approach to pattern recognition, enhancing its utility in forecasting and anomaly detection tasks. We followed the original paper's settings for this model in each dataset. GitHub: [CoST](https://github.com/salesforce/CoST)

- **Informer**: The Informer model, known for its efficiency in time series forecasting, is designed for long sequence time series forecasting, previously held the status of state-of-the-art (SOTA) on ETT datasets. Its implementation is accessible through the open-source repository available at GitHub: [Informer](https://github.com/zhouhaoyi/Informer2020). For each dataset reproduction setting, we followed the experiment settings described in our paper.

- **LogTrans**: LogTrans modifies the transformer model for long sequence forecasting by implementing a logarithmic distance-based self-attention mechanism. This method balances computational efficiency with the ability to capture long-range dependencies. We use a modified version of a third-party implementation. GitHub: [LogTrans](https://github.com/mlpotter/Transformer_Time_Series). The embedding vector size is set to 256, and the kernel size for casual convolutions is 9. We stack three layers for their Transformer.

- **TCN**: Temporal Convolutional Networks (TCN) are known for their simplicity and effectiveness in sequence modeling for time series data. They utilize causal convolutions and can handle variable-length input sequences. GitHub: [TCN](https://github.com/locuslab/TCN). We are using the model which is structured with ten residual blocks stacked atop our backbone architecture, each featuring a hidden size of 64. We have set the maximum number of training epochs to 100 and use a learning rate of 0.001. All other parameters are retained as the default values specified in our code.

## More Forecasting Results

As depicted in Figure 1, the comparative analysis conducted between the TSI, CoST, and TS2Vec models revealed that the TSI model demonstrated enhanced accuracy, especially within the sample index ranges of 10-20 and 30-40. The TSI model's predictions closely mirrored the actual observations, contrasting with the CoST and TS2Vec models, which exhibited greater deviation at the data's extremities, thereby diminishing their precision. The TSI model's responsiveness was notably effective, swiftly adjusting to the data's fluctuations and signaling a more agile adaptation mechanism in comparison to the CoST model, which showed a tangible lag in its response to the time series' directional shifts.

In Figure 2, a comprehensive comparative analysis conducted on the ETTh1 dataset with 500 sample indices indicated that the TSI model displayed superior accuracy, particularly in the sample index ranges of 10-20 and 30-40. This model's predictions were closely aligned with actual observations, outperforming the CoST and TS2Vec models, which showed greater deviation at extreme data points. The TSI model's responsiveness to fluctuations in the data was remarkable, indicating a more agile adaptation mechanism when compared to the CoST model, which revealed a noticeable lag in response to directional shifts in the time series.

<img width="548" alt="截屏2024-06-18 下午11 04 25" src="https://github.com/Wentao-Gao/TSI-forcasting/assets/77952995/37e6f7cd-6d30-40c3-89f5-eb1d9c869ca5">


<img width="525" alt="截屏2024-06-18 下午11 04 47" src="https://github.com/Wentao-Gao/TSI-forcasting/assets/77952995/218f70e6-ee41-41bb-874f-7f408df78464">



