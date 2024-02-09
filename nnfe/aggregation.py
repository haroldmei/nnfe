import warnings
warnings.filterwarnings('ignore')

import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import copy
import traceback

from joblib import Parallel, delayed

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import minmax_scale
from tqdm import tqdm_notebook as tqdm

from neighbor import TimeIdNeighbor, EntityIdNeighbor, Neighbor, MAX_ENTITYID_NEIGHBORS, MAX_TIMEID_NEIGHBORS
from enum import Enum
from typing import Dict, List, Optional, Tuple

def print_trace(name: str = ''):
    print(f'ERROR RAISED IN {name or "anonymous"}')
    print(traceback.format_exc())

def tid_neighbor(df, vals, metric='minowski', num_neibors=MAX_TIMEID_NEIGHBORS):
    """
    Load time-id neighbor based on the given dataframe and parameters.

    Args:
        df (pandas.DataFrame): The input dataframe.
        vals (str): The column name of the values to be used for neighbor calculation.
        metric (str, optional): The distance metric to be used for neighbor calculation. Defaults to 'minowski'.
        num_neibors (int, optional): The number of neighbors to be considered. Defaults to MAX_TIMEID_NEIGHBORS.

    Returns:
        TimeIdNeighbor: The time-id neighbor object.

    Raises:
        ValueError: If the given metric is not supported.
    """
    pivot = df.pivot(index='time_id', columns='entity_id', values=vals)
    pivot = pivot.fillna(pivot.mean())
    pivot = pd.DataFrame(minmax_scale(pivot))
    
    if metric == 'canberra':
        time_id_neighbor = TimeIdNeighbor(f"{'-'.join(vals)}-canberra-tid", pivot, p=2, metric=metric, exclude_self = True, num_neibors=num_neibors)
    elif metric == 'mahalanobis':
        time_id_neighbor = TimeIdNeighbor(f"{'-'.join(vals)}-mahalanobis-tid", pivot, p=2, metric=metric, metric_params = {'VI': np.cov(pivot.values.T)}, num_neibors=num_neibors)
    elif metric == 'minowski':
        time_id_neighbor = TimeIdNeighbor(f"{'-'.join(vals)}-minowski-tid", pivot, p=2, metric=metric, num_neibors=num_neibors)
    else:
        raise ValueError(f'unsupported metric {metric}')
    
    time_id_neighbor.generate_neighbors()
    return time_id_neighbor


def eid_neighbor(df, vals, metric='minkowski', num_neibors=MAX_ENTITYID_NEIGHBORS):
    """
    Load entity-id neighbor based on the given dataframe and parameters.

    Args:
        df (pandas.DataFrame): The input dataframe.
        vals (str): The column name of the values to be used for neighbor calculation.
        metric (str, optional): The distance metric to be used for neighbor calculation. Defaults to 'minkowski'.
        num_neibors (int, optional): The number of neighbors to be considered. Defaults to MAX_ENTITYID_NEIGHBORS.

    Returns:
        EntityIdNeighbor: The entity-id neighbor object.

    Raises:
        ValueError: If the given metric is not supported.
    """
    pivot = df.pivot(index='time_id', columns='entity_id', values=vals)
    pivot = pivot.fillna(pivot.mean())
    pivot = pd.DataFrame(minmax_scale(pivot))
    entity_id_neighbor = EntityIdNeighbor(f"{'-'.join(vals)}-{metric}-eid", pivot, p=1, metric=metric, exclude_self=True, num_neibors=num_neibors)

    entity_id_neighbor.generate_neighbors()
    return entity_id_neighbor


def eid_dirichlet_emb(df, vals, n_components=3):
    """
    Load the Dirichlet allocation based on the given dataframe and parameters.

    Args:
        df (pandas.DataFrame): The input dataframe.
        vals (str): The column name of the values to be used for Dirichlet allocation.
        n_components (int, optional): The number of components to be used for Dirichlet allocation. Defaults to 16.

    Returns:
        pd.DataFrame: The DataFrame containing the Dirichlet allocation results.
    """
    lda = LatentDirichletAllocation(n_components=n_components, random_state=0)
    pivot = df.pivot(index='entity_id', columns='time_id', values=vals)
    lda_df = pd.DataFrame(lda.fit_transform(pivot), index=pivot.columns)
    lda_df = pd.DataFrame(lda.fit_transform(pivot.transpose()), index=pivot.columns)
    for i in range(n_components):
        pivot[f'entity_id_emb{i}'] = pivot['time_id'].map(lda_df[i])
    return pivot



def tid_dirichlet_emb(df, vals, n_components=3):
    """
    Load the Dirichlet allocation based on the given dataframe and parameters.

    Args:
        df (pandas.DataFrame): The input dataframe.
        vals (str): The column name of the values to be used for Dirichlet allocation.
        n_components (int, optional): The number of components to be used for Dirichlet allocation. Defaults to 16.

    Returns:
        pd.DataFrame: The DataFrame containing the Dirichlet allocation results.
    """
    lda = LatentDirichletAllocation(n_components=n_components, random_state=0)
    pivot = df.pivot(index='time_id', columns='entity_id', values=vals)
    lda_df = pd.DataFrame(lda.fit_transform(pivot.transpose()), index=pivot.columns)
    for i in range(n_components):
        pivot[f'time_id_emb{i}'] = pivot['entity_id'].map(lda_df[i])
    return pivot

def load_neighbors(df):
    tid_neighbors: List[Neighbor] = []
    eid_neigibors: List[Neighbor] = []

    try:
        # keep the original dataframes
        tid_neighbors.append(tid_neighbor(df, ['close'], metric='canberra'))
        eid_neigibors.append(eid_neighbor(df, ['close'], metric='canberra'))

        print('@@@@@ Start fitting neighbors @@@@@')
        def generate_neighbors(n):
            n.generate_neighbors()
            return n
        
        all_neibors = Parallel(n_jobs=-1)(delayed(generate_neighbors)(n) for n in tid_neighbors + eid_neigibors)
        tid_neighbors = all_neibors[:len(tid_neighbors)]
        eid_neigibors = all_neibors[len(tid_neighbors):]

    except Exception:
        print_trace('load_neighbors')
        exit(1)

    return tid_neighbors, eid_neigibors

def _add_ndf(ndf: Optional[pd.DataFrame], dst: pd.DataFrame) -> pd.DataFrame:
    """
    Add a new DataFrame to an existing DataFrame.

    Args:
        ndf (Optional[pd.DataFrame]): The new DataFrame to be added. If None, no addition is performed.
        dst (pd.DataFrame): The existing DataFrame to which the new DataFrame is added.

    Returns:
        pd.DataFrame: The resulting DataFrame after the addition.

    """
    if ndf is None:
        for col in dst.columns:
            if col in ['time_id', 'entity_id']:
                continue
            dst[col] = dst[col].astype(np.float32)
        return dst
    else:
        ndf[dst.columns[-1]] = dst[dst.columns[-1]].astype(np.float32)
        return ndf
    

def nn_feature(df, feature_col, aggs, neighbors, neighbor_sizes):
    """
    Generate nearest neighbor features based on the given DataFrame, feature column, aggregation methods,
    neighbor objects, and neighbor sizes.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        feature_col (str): The name of the column to be used as the feature.
        aggs (list): A list of aggregation methods to be applied.
        neighbors (list): A list of neighbor objects.
        neighbor_sizes (list): A list of neighbor sizes.

    Returns:
        list: A list of generated nearest neighbor features.
    """
    dsts = []
    try:
        if feature_col not in df.columns:
            print(f"column {feature_col} is skipped")
            return
        
        if not neighbors:
            return
        
        for nn in range(len(neighbors)):
            neighbor = copy.deepcopy(neighbors[nn])
            neighbor.rearrange_feature_values(df, feature_col)
            neighbors[nn] = neighbor

        for agg in aggs:
            for n in neighbor_sizes:
                for nn in neighbors:
                    dst = nn.make_nn_feature(n, agg)
                    dsts.append(dst)

    except Exception:
        print_trace('nn aggregation failed for {}, {}'.format(feature_col, aggs))
        exit(1)

    return dsts

def nn_features(df: pd.DataFrame, 
                  neighbors: List[Neighbor], 
                  feature_cols,
                  neigbor_sizes = [5]
                ) -> pd.DataFrame:
    """
    Generate nearest neighbor features for a given dataframe.

    Args:
        df (pd.DataFrame): The input dataframe.
        neighbors (List[Neighbor]): List of neighbor objects.
        feature_cols: Dictionary of feature columns.
        neigbor_sizes (List[int], optional): List of neighbor sizes. Defaults to [5].

    Returns:
        pd.DataFrame: The dataframe with nearest neighbor features added.
    """
    
    # make a copy of the original dataframe
    df2 = df.copy()

    ndf: Optional[pd.DataFrame] = None
    dsts = Parallel(n_jobs=16)(delayed(nn_feature)(
            df2, 
            feature_col, 
            feature_cols[feature_col], 
            neighbors, 
            neigbor_sizes
        ) for feature_col in feature_cols.keys()
    )

    for flat_list in dsts:
        if flat_list is None or len(flat_list) == 0:
            continue
        for dst in flat_list:
            ndf = _add_ndf(ndf, dst)
    
    if ndf is not None:
        df2 = pd.merge(df2, ndf, on=['time_id', 'entity_id'], how='left')

    return df2

def make_nn_feature(df):
    """
    Generate neural network features for the given dataframe.

    Parameters:
    df (pandas.DataFrame): The input dataframe.

    Returns:
    pandas.DataFrame: The dataframe with neural network features.
    """
    tid_neighbors, eid_neigibors = load_neighbors(df)
    df = nn_features(df, tid_neighbors, {'open_diff1': [np.mean, np.std]})
    df = df.reset_index(drop=True)
    return df
        

