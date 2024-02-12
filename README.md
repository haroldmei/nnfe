# nnfe: Nearest Neighbors Feature Extraction

nnfe is a useful machine learning tool for aggregating features from nearest neighbors for time series data. I saw this piece of code originally from kaggle competition [Optiver Realized Volatility Prediction](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/274970) where kaggle grandmaster [nyanp](https://www.kaggle.com/nyanpn) use this tool and won the first prize in the contest. The competition was fun, it has data leakage problem in the final test data, and the nearest neighbor feature extraction helped nyanp to exploit leaked data to stand out; and the problem dataset is extremely noisy so the problem remains unsolved. But the idea of using nearest neighbors to either extract features or simply as a part of the ensembling model is obviously very impressive to me, so I made some improvements to the code (especially add support to hide future data in neighbor generation to avoid data leakage), package it and made a lib so that it can be easily utilized by the data science community.

## Quick start
### Installation
### Use from CLI
### Use as python lib

## Introduction

Suppose we have the following time series data, which is the stock prices of AMZN, AAPL, GOOG and TSLA from 2024.02.01 ~ 2024.02.09.

| Date        | Symbol      | Open        | High        | Low        | Close      
| ----------- | ----------- | ----------- | ----------- | -----------| -----------  
| 2024.02.01  | AMZN        | 157.14      | 159.30      | 156.68     | 158.66       
| 2024.02.02  | AMZN        | 169.75      | 172.46      | 168.95     | 171.83       
| 2024.02.05  | AMZN        | 169.28      | 170.35      | 167.68     | 170.07       
| 2024.02.06  | AMZN        | 170.28      | 170.40      | 167.63     | 168.72       
| 2024.02.07  | AMZN        | 169.97      | 170.86      | 169.14     | 170.33       
| 2024.02.08  | AMZN        | 169.08      | 171.39      | 169.05     | 169.75       
| 2024.02.09  | AMZN        | 171.22      | 174.97      | 170.56     | 174.36       
| 2024.02.01  | AAPL        | 185.01      | 186.88      | 184.57     | 186.57       
| 2024.02.02  | AAPL        | 181.53      | 187.13      | 181.40     | 186.62       
| 2024.02.05  | AAPL        | 187.21      | 189.22      | 185.83     | 188.09       
| 2024.02.06  | AAPL        | 187.86      | 189.10      | 187.26     | 188.78      
| 2024.02.07  | AAPL        | 190.04      | 190.38      | 188.59     | 189.19      
| 2024.02.08  | AAPL        | 188.85      | 189.50      | 187.33     | 188.13       
| 2024.02.09  | AAPL        | 188.41      | 189.97      | 188.21     | 188.71       
| 2024.02.01  | GOOG        | 143.88      | 144.09      | 142.24     | 142.68      
| 2024.02.02  | GOOG        | 139.06      | 143.86      | 138.70     | 143.66      
| 2024.02.05  | GOOG        | 145.56      | 146.64      | 144.51     | 145.05       
| 2024.02.06  | GOOG        | 146.09      | 146.32      | 144.50     | 145.15       
| 2024.02.07  | GOOG        | 145.98      | 146.98      | 145.17     | 146.65       
| 2024.02.08  | GOOG        | 147.19      | 147.54      | 146.40     | 147.12       
| 2024.02.09  | GOOG        | 148.03      | 150.68      | 147.85     | 150.09       
| 2024.02.01  | TSLA        | 188.64      | 189.76      | 184.21     | 188.24       
| 2024.02.02  | TSLA        | 184.77      | 188.63      | 182.16     | 188.02      
| 2024.02.05  | TSLA        | 183.82      | 184.46      | 174.98     | 180.70      
| 2024.02.06  | TSLA        | 180.92      | 186.43      | 180.23     | 184.41       
| 2024.02.07  | TSLA        | 187.30      | 189.75      | 182.65     | 187.64       
| 2024.02.08  | TSLA        | 186.85      | 191.58      | 185.54     | 189.33      
| 2024.02.09  | TSLA        | 191.56      | 194.01      | 190.48     | 193.34      


## Scenario 1: Find nearest neighbors from history

On 2024.02.09, when the market just opened. The open prices of the four companies are available and are  represented as a price vector: 
```math
\vec{p}=(171.22, 188.41, 148.03, 191.56)
```

If we are able to find the nearest price vector for all these companies' open prices from history, It does not seem too crazy for us to use that day's close prices to predict close prices of 2024.02.09. One step further, as a normal practice in data science, we can take one step further to find k nearest neighbors and use an aggregated close price (such as take the mean) to predict 2024.02.09's close price.

Even if we don't think kNN is a good model to predict a day's close, the aggregated close price will be a great feature that can be used for a more powerful supervised learning model such as [LightGBM](https://lightgbm.readthedocs.io/en/stable/), or deep learning models.

In this scenario we use only open price to form a feature vector for only 4 companies, and use the 4 dimensional vector to find nearest neighbors. In real life we may be interested in using multiple features and form feature vectors for the SP500 companies, or even the entire NASDAQ plus NYSE companies; the dimension of the vectors formed will be hundreds to thousands, the computation of distances between super high dimension vectors will soon be bottlenecks of the system. So it is necessary to include different types of vector distances, for the purpose of fast computation plus sufficient information for distance comparison.


## Scenario 2: Find most similar entities 

The second scenario is to use an entity's feature time series to find the most similar entities from the collection of entities. In the above case an entity is a company's symbol. Suppose we use AMZN's open price series as the price vector:
```math
\vec{p}=(157.14, 169.75, 169.28, 170.28, 169.97, 169.08, 171.22)
```

If we can find the k most similar companies based on that companies open price series, we can again aggregate (such as take the mean of) the close prices on each day and include this new feature in the model training.

## Generalize the feature extraction steps

Given a time series data with the columns (eid, tid, feature1, feature2, ...), suppose the data is aligned to $M$ entities and $N$ time steps. By making use of the following steps we can extract some very useful features.

* Specify a list of observasion features that will be used to form feature vectors and generate nearest neighbors.
  ```
  from nnfe.aggregation import tid_neighbor, eid_neighbor, nn_features
  
  # find historical nearest neighbors by observing 'open' feature
  tid_neighbors: List[Neighbor] = []
  n = tid_neighbor(df, ['open'], metric='canberra')
  n.generate_neighbors()
  tid_neighbors.append(n)

  # find most similar entity by observing 'open' feature
  eid_neigibors: List[Neighbor] = []
  n = eid_neigibor(df, ['open'], metric='canberra')
  n.generate_neighbors()
  eid_neigibors.append(eid_neighbor(df, ['open'], metric='canberra'))

* Specify a list of features and corresponding aggregations to aggregation features from the nearest neighbors. The aggregation can be defined differently as a function.
  ```
  # aggregate features for 'close' feature from nearest neighbors using np.mean and np.std
  agg = [ {'close': [np.mean, np.std]} ]
  df = nn_features(df, tid_neighbors, agg)


