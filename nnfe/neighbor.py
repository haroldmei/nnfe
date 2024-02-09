import warnings
warnings.filterwarnings('ignore')

import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import copy

from joblib import Parallel, delayed

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import minmax_scale
from tqdm import tqdm_notebook as tqdm

from enum import Enum
from typing import Dict, List, Optional, Tuple

MAX_ENTITYID_NEIGHBORS = 15 # for entity_id aggregated neighbors
MAX_TIMEID_NEIGHBORS = 40 # for time_id aggregated neighbors

class Neighbor:
    def __init__(self, 
                 name: str, 
                 pivot: pd.DataFrame, 
                 time_id: str, 
                 entity_id: str,
                 p: float, 
                 metric: str = 'minkowski', 
                 metric_params: Optional[Dict] = None, 
                 exclude_self: bool = True, #False, hmei
                 num_neibors = MAX_ENTITYID_NEIGHBORS):
        """
        Initializes a Neighbor object.

        Args:
            name (str): The name of the Neighbor.
            pivot (pd.DataFrame): The pivot table used for generating neighbors.
            p (float): The power parameter for the Minkowski metric.
            metric (str, optional): The distance metric to use. Defaults to 'minkowski'.
            metric_params (Optional[Dict], optional): Additional parameters for the distance metric. Defaults to None.
            exclude_self (bool, optional): Whether to exclude the self entity from the neighbors. Defaults to True.
            num_neibors (int, optional): The number of neighbors to generate. Defaults to MAX_ENTITYID_NEIGHBORS.
        """
        self.name = name
        self.time_id = time_id
        self.entity_id = entity_id
        self.exclude_self = exclude_self
        self.p = p
        self.metric = metric
        self.n_numbers = num_neibors
        self.columns = self.index = self.feature_values = self.feature_col = None
        self.metric_params = metric_params
        self.pivot = pivot
        self.neighbors = None

    def generate_neighbors(self):
        """
        Generates the neighbors based on the specified parameters.
        """
        if self.metric == 'random':
            n_queries = len(self.pivot)
            self.neighbors = np.random.randint(n_queries, size=(n_queries, self.n_numbers))
        else:
            nn = NearestNeighbors(
                n_neighbors=self.n_numbers, 
                p=self.p, 
                metric=self.metric, 
                metric_params=self.metric_params
            )
            nn.fit(self.pivot)
            _, self.neighbors = nn.kneighbors(self.pivot, return_distance=True)
            #print(self.neighbors)


    def rearrange_feature_values(self, df: pd.DataFrame, feature_col: str) -> None:
        """
        Rearranges the feature values of the Neighbor.

        Args:
            df (pd.DataFrame): The DataFrame containing the feature values.
            feature_col (str): The name of the feature column.
        """
        raise NotImplementedError()

    def make_nn_feature(self, n=5, agg=np.mean) -> pd.DataFrame:
        """
        Generates the nearest neighbor feature.

        Args:
            n (int, optional): The number of neighbors to consider. Defaults to 5.
            agg (function, optional): The aggregation function to apply. Defaults to np.mean.

        Returns:
            pd.DataFrame: The DataFrame containing the nearest neighbor feature.
        """
        assert self.feature_values is not None, "should call rearrange_feature_values beforehand"

        start = 1 if self.exclude_self else 0
        pivot_aggs = pd.DataFrame(
            agg(self.feature_values[start:n,:,:], axis=0), 
            columns=self.columns, 
            index=self.index
        )

        dst = pivot_aggs.unstack().reset_index()
        dst.columns = [self.entity_id, self.time_id, f'{self.feature_col}_knn_{n}_{self.name}_{agg.__name__}']
        return dst

class TimeIdNeighbor(Neighbor):
    def rearrange_feature_values(self, df: pd.DataFrame, feature_col: str) -> None:
        """
        Rearranges the feature values based on the neighbors.

        Args:
            df (pd.DataFrame): The input DataFrame.
            feature_col (str): The name of the feature column.

        Returns:
            None
        """
        # filter future data first for time series, retain the original order
        for i in range(self.neighbors.shape[0]):
            self.neighbors.setflags(write=1)
            if self.metric == 'random':
                self.neighbors[i, :] = np.where(self.neighbors[i,:] < i, self.neighbors[i,:], i-1 if i > 0 else 0) # a hack, makesure the item itself is not choosen for feature aggregation
            else:
                self.neighbors[i, :] = np.where(self.neighbors[i,:] < i, self.neighbors[i,:], i)
            _, idx = np.unique(self.neighbors[i, :], return_index=True)
            self.neighbors[i, :idx.shape[0]] = self.neighbors[i, np.sort(idx)]
            self.neighbors[i, idx.shape[0]:] = self.neighbors[i, np.sort(idx)[-1]] #i

        feature_pivot = df.pivot(index=self.time_id, columns=self.entity_id, values=feature_col)
        feature_pivot = feature_pivot.fillna(feature_pivot.mean())
        feature_pivot.head()

        feature_values = np.zeros((self.n_numbers, *feature_pivot.shape))

        for i in range(self.n_numbers):
            feature_values[i, :, :] += feature_pivot.values[self.neighbors[:, i], :]

        self.columns = list(feature_pivot.columns)
        self.index = list(feature_pivot.index)
        self.feature_values = feature_values
        self.feature_col = feature_col
        
        #print(self.neighbors.shape, type(self.neighbors))
        #df = pd.DataFrame({'neibors': self.neighbors.tolist()})
        #df.to_csv(f'{DATA_DIR}/debug/{self.name}.csv')

    def __repr__(self) -> str:
        """
        Returns a string representation of the TimeIdNeighbor object.

        Returns:
            str: The string representation of the object.
        """
        return f"time-id NN (name={self.name}, metric={self.metric}, p={self.p})"

class EntityIdNeighbor(Neighbor):
    def rearrange_feature_values(self, df: pd.DataFrame, feature_col: str) -> None:
        """Rearranges feature values based on entity-id nearest neighbors.

        Args:
            df (pd.DataFrame): The input DataFrame.
            feature_col (str): The name of the feature column.

        Returns:
            None
        """
        feature_pivot = df.pivot(index=self.time_id, columns=self.entity_id, values=feature_col)
        feature_pivot = feature_pivot.fillna(feature_pivot.mean())
        feature_values = np.zeros((MAX_ENTITYID_NEIGHBORS, *feature_pivot.shape))
        for i in range(MAX_ENTITYID_NEIGHBORS):
            feature_values[i, :, :] += feature_pivot.values[:, self.neighbors[:, i]]

        self.columns = list(feature_pivot.columns)
        self.index = list(feature_pivot.index)
        self.feature_values = feature_values
        self.feature_col = feature_col
        
    def __repr__(self) -> str:
        return f"entity-id NN (name={self.name}, metric={self.metric}, p={self.p})"

