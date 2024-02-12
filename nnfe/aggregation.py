import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import copy
import traceback

from joblib import Parallel, delayed

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import minmax_scale

from nnfe.neighbor import TimeIdNeighbor, EntityIdNeighbor, Neighbor, MAX_ENTITYID_NEIGHBORS, MAX_TIMEID_NEIGHBORS
from typing import List, Optional

def print_trace(name: str = ''):
    print(f'ERROR RAISED IN {name or "anonymous"}')
    print(traceback.format_exc())

def tid_neighbor(df, vals, metric='minkowski', num_neibors=MAX_TIMEID_NEIGHBORS):
    """
    Load time-id neighbor based on the given dataframe and parameters.

    Args:
        df (pandas.DataFrame): The input dataframe.
        vals (str): The column name of the values to be used for neighbor calculation.
        metric (str, optional): The distance metric to be used for neighbor calculation. Defaults to 'minkowski'.
        num_neibors (int, optional): The number of neighbors to be considered. Defaults to MAX_TIMEID_NEIGHBORS.

    Returns:
        TimeIdNeighbor: The time-id neighbor object.

    Raises:
        ValueError: If the given metric is not supported.
    """
    time_id = df.columns[0]
    entity_id = df.columns[1]
    pivot = df.pivot(index=time_id, columns=entity_id, values=vals)
    pivot = pivot.fillna(pivot.mean())
    pivot = pd.DataFrame(minmax_scale(pivot))
    
    if metric == 'canberra':
        time_id_neighbor = TimeIdNeighbor(f"{'-'.join(vals)}-canberra-tid", 
                                        pivot, 
                                        time_id, 
                                        entity_id,
                                        p=2, 
                                        metric=metric, 
                                        exclude_self = True, 
                                        num_neibors=num_neibors
                                    )
    elif metric == 'mahalanobis':
        time_id_neighbor = TimeIdNeighbor(f"{'-'.join(vals)}-mahalanobis-tid", 
                                          pivot, 
                                        time_id, 
                                        entity_id,
                                          p=2, 
                                          metric=metric, 
                                          metric_params = {'VI': np.cov(pivot.values.T)}, 
                                          num_neibors=num_neibors
                                        )
    elif metric == 'minkowski':
        time_id_neighbor = TimeIdNeighbor(f"{'-'.join(vals)}-minkowski-tid", 
                                          pivot, 
                                        time_id, 
                                        entity_id,
                                          p=2, 
                                          metric=metric, 
                                          num_neibors=num_neibors
                                        )
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
    time_id = df.columns[0]
    entity_id = df.columns[1]
    pivot = df.pivot(index=time_id, columns=entity_id, values=vals)
    pivot = pivot.fillna(pivot.mean())
    pivot = pd.DataFrame(minmax_scale(pivot))
    entity_id_neighbor = EntityIdNeighbor(f"{'-'.join(vals)}-{metric}-eid", 
                                          pivot, 
                                        time_id, 
                                        entity_id,
                                          p=1, 
                                          metric=metric, 
                                          exclude_self=True, 
                                          num_neibors=num_neibors
                                        )

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
    
    time_id = df.columns[0]
    entity_id = df.columns[1]
    pivot = df.pivot(index=time_id, columns=entity_id, values=vals)

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
    
    time_id = df.columns[0]
    entity_id = df.columns[1]
    pivot = df.pivot(index=time_id, columns=entity_id, values=vals)
    
    lda_df = pd.DataFrame(lda.fit_transform(pivot.transpose()), index=pivot.columns)
    for i in range(n_components):
        pivot[f'time_id_emb{i}'] = pivot['entity_id'].map(lda_df[i])
    return pivot

def load_neighbors(df, features, type='tid'):
    """
    Load neighbors based on the given dataframe and features.

    Args:
        df (pandas.DataFrame): The dataframe containing the data.
        features (list): The list of features to consider for neighbor generation.
        type (str, optional): The type of neighbors to load. Defaults to 'tid'.

    Returns:
        list: The list of loaded neighbors.
    """
    neighbors = []
    if type not in ['tid', 'eid']:
        raise ValueError(f'unsupported type {type}')
    try:
        # keep the original dataframes
        neighbors.append(
            tid_neighbor(df, features, metric='canberra') if type == 'tid' else eid_neighbor(df, features, metric='canberra')
        )

        print('@@@@@ Start fitting neighbors @@@@@')
        def generate_neighbors(n):
            n.generate_neighbors()
            return n
        
        all_neibors = Parallel(n_jobs=-1)(delayed(generate_neighbors)(n) for n in neighbors)
        neighbors = all_neibors
    except Exception:
        print_trace('load_neighbors')
        exit(1)

    return neighbors

def _add_ndf(ndf: Optional[pd.DataFrame], dst: pd.DataFrame) -> pd.DataFrame:
    """
    Add a new DataFrame to an existing DataFrame.

    Args:
        ndf (Optional[pd.DataFrame]): The new DataFrame to be added. If None, no addition is performed.
        dst (pd.DataFrame): The existing DataFrame to which the new DataFrame is added.

    Returns:
        pd.DataFrame: The resulting DataFrame after the addition.

    """

    time_id = dst.columns[0]
    entity_id = dst.columns[1]
    if ndf is None:
        for col in dst.columns:
            if col in [time_id, entity_id]:
                continue
            dst[col] = dst[col].astype(np.float32)
        return dst
    else:
        ndf[dst.columns[-1]] = dst[dst.columns[-1]].astype(np.float32)
        return ndf
    

def nn_feature(df, feature_col, aggs, neighbors, neighbor_sizes, leakage=False):
    """
    Generate nearest neighbor features based on the given DataFrame, feature column, aggregation methods,
    neighbor objects, and neighbor sizes.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        feature_col (str): The name of the column to be used as the feature.
        aggs (list): A list of aggregation methods to be applied.
        neighbors (list): A list of neighbor objects.
        neighbor_sizes (list): A list of neighbor sizes.
        leakage (bool, optional): Flag indicating whether to consider leakage. Defaults to False.

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
            neighbor.rearrange_feature_values(df, feature_col, leakage=leakage)
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
                  neigbor_sizes = [5],
                  leakage=False # Disallow data leakage
                ) -> pd.DataFrame:
    """
    Generate nearest neighbor features for a given dataframe.

    Args:
        df (pd.DataFrame): The input dataframe.
        neighbors (List[Neighbor]): List of neighbor objects.
        feature_cols: Dictionary of feature columns.
        neigbor_sizes (List[int], optional): List of neighbor sizes. Defaults to [5].
        leakage (bool, optional): Flag to disallow data leakage. Defaults to False.

    Returns:
        pd.DataFrame: The dataframe with nearest neighbor features added.
    """
    
    # make a copy of the original dataframe
    df2 = df.copy()
    time_id = df.columns[0]
    entity_id = df.columns[1]

    ndf: Optional[pd.DataFrame] = None
    dsts = Parallel(n_jobs=16)(delayed(nn_feature)(
            df2, 
            feature_col, 
            feature_cols[feature_col], 
            neighbors, 
            neigbor_sizes,
            leakage=leakage
        ) for feature_col in feature_cols.keys()
    )

    for flat_list in dsts:
        if flat_list is None or len(flat_list) == 0:
            continue
        for dst in flat_list:
            ndf = _add_ndf(ndf, dst)
    
    if ndf is not None:
        df2 = pd.merge(df2, ndf, on=[time_id, entity_id], how='left')

    return df2

def make_nn_feature(df, eid, tid, target, feature, aggregation, leakage=False, nntype='tid'):
    """
    Generate neural network features for the given dataframe.

    Parameters:
    df (pandas.DataFrame): The input dataframe.
    eid (str): The column name representing the entity ID.
    tid (str): The column name representing the target ID.
    target (str): Comma-separated string of column names representing the target variables.
    feature (str): The column name representing the feature to be aggregated.
    aggregation (str): The aggregation function to be applied to the feature.
    leakage (bool, optional): Whether to include leakage prevention in the feature generation. Defaults to False.
    nntype (str, optional): The type of nearest neighbors to consider. Defaults to 'tid'.

    Returns:
    pandas.DataFrame: The dataframe with neural network features.
    """
    cols = [tid, eid] + [i for i in df.columns if i not in [eid, tid]]
    df = df[cols]

    targets = target.split(',')
    for t in targets:
        if t not in df.columns:
            raise ValueError(f'{t} is not in the dataframe')

    neighbors = load_neighbors(df, targets, nntype=nntype)
    df = nn_features(df, neighbors, {feature: [aggregation]}, leakage=leakage)
    df = df.reset_index(drop=True)
    return df
        

