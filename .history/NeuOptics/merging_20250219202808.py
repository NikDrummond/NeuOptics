from numpy import vstack, log, ndarray, unique, where
from warnings import warn
from copy import deepcopy
from scipy.spatial import KDTree
from typing import List
from functools import reduce
from Neurosetta import Forest_graph, 

from .columns import Columns

def add_column(cols1:Columns,cols2:Columns, ind1:int, ind2:int, force_centroid_update:bool = False):
    """Add a single column from cols2 to a single column in cols1

    Parameters
    ----------
    cols1 : Columns
        Columns for column to be merged into
    cols2 : Columns
        Columns for column to be merged from
    ind1 : int
        Index of column to be merged into
    ind2 : int
        index of column to be merged from
    force_centroid_update : bool, optional
        If True, forced the update of all column centres in cols1, by default False
    """
    
    # extend lists for col1[ind1]
    cols1.Column_points[ind1] = vstack((cols1.Column_points[ind1],cols2.Column_points[ind2]))
    cols1.Column_types[ind1].extend(cols2.Column_types[ind2])
    cols1.Column_N_ids[ind1].extend(cols2.Column_N_ids[ind2])
    cols1.Synapse_counts[ind1].extend(cols2.Synapse_counts[ind2])
    if force_centroid_update:
        cols1.update_centres()

    return cols1

def _outlier_detection(dist,threshold: float = 5000, outlier_method:str = 'absolute') -> ndarray:
    accepter_outlier_methods = ['absolute', 'z_score']

    if outlier_method == 'absolute':
        outlier_mask = (dist > threshold)
    elif outlier_method == 'z_score':
        if threshold > 5:
            warn(f"Outlier threshold is set to {threshold} standard deviations, this is likely to high. \n are you sure you do not mean to set outlier_method to 'absolute'?")
        # lets z score the log distances to look for outliers
        data = log(dist)
        mean = data.mean()
        std = data.std()
        outlier_mask = (abs(data - mean) > threshold * std)
    else:
        raise ValueError(f'Given outlier detection method {outlier_method} is not valid. Expected one of {accepter_outlier_methods}')
    return outlier_mask

def merge_columns(
    target_columns: Columns,
    source_columns: Columns,
    remove_outliers: bool = True,
    threshold: float = 5000,
    outlier_method: str = "absolute",
    force_unique: bool = False,
) -> Columns:
    """Merge Columns from source_columns into Columns in target_columns based on minimum distance between column centres

    Parameters
    ----------
    target_columns : Columns
        Columns to merge into
    source_columns : Columns
        Columns to merge from
    remove_outliers : bool, optional
        Should we try to remove outliers?, by default True
    threshold : float, optional
        threshold value for outlier removal, by default 5000
    outlier_method : str, optional
        method for outlier detection. Can be either absolute or z_score.
        If absolute, column merges greater than this value (in spatial units of points which make up the columns)
        will not be merged.
        
        Alternatively, if method is z_score, this value is the number of standard deviations away from the mean of the 
        log distances between columns. 
        By default "absolute"
    force_unique : bool, optional
        If True, only one column from source_columns can be merged into one column from target_columns, by default False

    Returns
    -------
    Columns
        Copy of target_columns with columns from source_columns merged into it
    """
    cols1 = deepcopy(target_columns)
    # make a tree of the first set of columns
    tree = KDTree(cols1.Column_centres)
    # get nearest neighbour distances of second set and first set
    dist, ind = tree.query(source_columns.Column_centres, k=1)

    # generate a mask to use for outliers
    if remove_outliers:
        outlier_mask = _outlier_detection(dist, threshold, outlier_method)

    # merge!

    # get unique inds and counts
    uni_inds, counts = unique(ind, return_counts=True)

    # iterate over unique inds -
    for i in range(len(uni_inds)):
        # what we are thinking of merging into
        ind1 = uni_inds[i]
        # if there is only one
        if counts[i] == 1:
            # ind of point we want to add
            ind2 = where(ind == ind1)[0][0]
            # if we want to remove outliers
            if remove_outliers:
                # check if this is an outlier
                if outlier_mask[where(ind == ind1)]:
                    # skip this completely
                    continue
                else:
                    cols1 = add_column(cols1, source_columns, ind1, ind2)
            else:
                cols1 = add_column(cols1, source_columns, ind1, ind2)
        # if we have multiple contenders to merge into
        elif counts[i] > 1:
            # what are the possible sources
            possible_merges = where(ind == ind1)[0]

            # if we are forcing unique, which of these is closer?
            if force_unique:
                # get whichever one is closer
                ind2 = possible_merges[dist[possible_merges].argmin()]
                # are we removing outliers?
                if remove_outliers:
                    if outlier_mask[ind2]:
                        continue
                    else:
                        cols1 = add_column(cols1, source_columns, ind1, ind2)
                else:
                    cols1 = add_column(cols1, source_columns, ind1, ind2)
            # otherwise iterate through all possible merges
            else:
                for ind2 in possible_merges:
                    # are we removing outliers?
                    if remove_outliers:
                        if outlier_mask[ind2]:
                            continue
                        else:
                            cols1 = add_column(cols1, source_columns, ind1, ind2)
                    else:
                        cols1 = add_column(cols1, source_columns, ind1, ind2)

    # update column centres
    cols1.update_centres()

    return cols1

def _get_data_for_cols(N_all: nr.Forest_graph, unicolumnar_input_types: List | str):
    """helper function to generate data we need to make columns from a Forest"""

    # generate synapse input table from Forest
    in_table = nr.Neuron_synapse_table(N_all, direction="inputs")
    # if we have only been given one type of input neuron
    if isinstance(unicolumnar_input_types, str):
        # turn into a list
        unicolumnar_input_types = [unicolumnar_input_types]

    # collect our data into dictionaries
    # get unique ids for each wanted input type - dict
    unique_id_dict = {
        t: np.unique(in_table.loc[in_table.Partner_type == t]["pre"].values)
        for t in unicolumnar_input_types
    }

    ### update here to make dict of all points, then take means for each.
    # get center of mass for each neuron synapse input type - dict of arrays
    all_pnts_dict = {
        t: [
            in_table.loc[in_table.pre == i][["graph_x", "graph_y", "graph_z"]].values
            for i in unique_id_dict[t]
        ]
        for t in unicolumnar_input_types
    }

    # find order going from most to least
    counts = np.array([len(unique_id_dict[t]) for t in unicolumnar_input_types])
    # sort input types
    sorted_input_types = [unicolumnar_input_types[i] for i in np.argsort(counts)[::-1]]

    return all_pnts_dict, unique_id_dict, sorted_input_types


def _make_columns(
    curr_type: str, all_pnts_dict: dict, unique_id_dict: dict
) -> NeuOptics.Columns:


    coords = np.array([i.mean(axis=0) for i in all_pnts_dict[curr_type]])
    col_ids = np.arange(coords.shape[0])
    col_point_coords = {i: all_pnts_dict[curr_type][i] for i in col_ids}
    types_in_columns = {i: [curr_type] for i in col_ids}
    ids_in_columns = {i: [unique_id_dict[curr_type][i]] for i in col_ids}
    Synapse_counts = {i: [all_pnts_dict[curr_type][i].shape[0]] for i in col_ids}

    return NeuOptics.Columns(
        coords,
        col_ids,
        col_point_coords,
        types_in_columns,
        ids_in_columns,
        Synapse_counts,
    )


def make_column_map(
    N_all: nr.Forest_graph,
    unicolumnar_input_types: List | str,
    remove_outliers: bool = True,
    threshold: float = 5000,
    outlier_method: str = "absolute",
    force_unique: bool = False,
) -> Columns:
    """_summary_

    Parameters
    ----------
    N_all : nr.Forest_graph
        Forest_grapyh of neurons where each neuron has input synapses and input partner neuron types
    unicolumnar_input_types : List | str
        Either single (str) or list of desired unicolumnar input types to make columns from
    remove_outliers : bool, optional
        Should we try to remove outliers?, by default True
    threshold : float, optional
        threshold value for outlier removal, by default 5000
    outlier_method : str, optional
        method for outlier detection. Can be either absolute or z_score.
        If absolute, column merges greater than this value (in spatial units of points which make up the columns)
        will not be merged.
        
        Alternatively, if method is z_score, this value is the number of standard deviations away from the mean of the 
        log distances between columns. 
        By default "absolute"
    force_unique : bool, optional
        If True, only one column from source_columns can be merged into one column from target_columns, by default False


    Returns
    -------
    NeuOptics.Columns
        Columns object with merged columns from all given unicolumnar neuron types
    """

    # get data we need to make Columns objects
    all_pnts_dict, unique_id_dict, sorted_input_types = _get_data_for_cols(
        N_all, unicolumnar_input_types
    )

    # we only have one to return
    if isinstance(unicolumnar_input_types, str):
        return _make_columns(unicolumnar_input_types, all_pnts_dict, unique_id_dict)

    # otherwise
    # make a dictionary of columns
    cols_dict = {
        i: _make_columns(i, all_pnts_dict, unique_id_dict) for i in sorted_input_types
    }

    # fixed function arguments
    arg1, arg2, arg3, arg4 = remove_outliers, threshold, outlier_method, force_unique
    # merge them
    # Use reduce to iteratively apply merge_columns
    return reduce(lambda v1, v2: merge_columns(v1, v2, arg1, arg2, arg3, arg4), 
                        (cols_dict[col] for col in sorted_input_types))  
    