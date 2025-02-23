from pandas import DataFrame
from .columns import Columns
from .flywire import flywire_column_assignment_table
import numpy as np
from typing import List
from warnings import warn
from scipy.optimize import linear_sum_assignment


def assign_preallocated_columns(cols:Columns,data:DataFrame | None = None, bind = True) -> None | dict:
    """Assign column IDs from a dataframe to columns object based on Columns_N_ids in given object and 
    lookup table data frame.

    Parameters
    ----------
    cols : NeuOptics.Columns
        A Columns object with your columns. Must have Columns_N_ids which can be looked up in data
    data : pd.DataFrame | None
        Provided pd.DataFrame with 'root_id', which match Columns_N_ids, and 'column_id' in it's columns, 
        to use as look up for column assignment. If None (Default) will assume the use of the flywire columns.
        Unassigned columns are given a value of -1
    bind : bool, optional
        If true, will add a dictionary to the attribute Assigned_columns in cols. Otherwise return a dictionary
        of columns_id (in given cols) : list(column ids for each N_id from data), by default True

    Returns
    -------
    None | dict
        id bind is True, adds attribute in place. otherwise returns dictionary
    """
    # if no data is given, assume we are using flywire
    if data is None:
        data = flywire_column_assignment_table()
    
    # initialise dictionary
    col_dict = dict() 
    # iterate over each columns, and create an attribute which maps N_ids to columns ids
    for col_id in cols.Column_N_ids:
        ids = cols.Column_N_ids[col_id]
        assigned_columns = []
        # iterate over ids (we do this so we can allocate -1 as not assigned)
        for i in ids:
            try:
                assigned_columns.append(data.loc[data.root_id == i,'column_id'].values[0])
            except:
                assigned_columns.append(-1)
        col_dict[col_id] = np.array(assigned_columns)

    if bind:
        cols.Assigned_columns = col_dict
    else:
        return col_dict
    
def _make_consensus(col: int, col_ids: np.ndarray, syn_counts: np.ndarray) -> int:
    """
    Internal function to determine the consensus column ID for a given column.

    Parameters
    ----------
    col : int
        The column index being processed.
    col_ids : np.ndarray
        Array of assigned column IDs, which may include -1 for unassigned values.
    syn_counts : np.ndarray
        Array of synapse counts corresponding to `col_ids`.

    Returns
    -------
    int
        The consensus column ID based on the highest synapse count. Returns:
        - The unique column ID if only one is present.
        - The remaining column ID after removing unassigned (-1) values.
        - The column ID with the highest synapse count if multiple remain.
        - -2 if a tie occurs for the highest synapse count.
    """
    if isinstance(syn_counts, List):
        syn_counts = np.array(syn_counts)
    if isinstance(col_ids, List):
        col_ids = np.array(col_ids)

    ### logic

    # if we have only one value
    if len(np.unique(col_ids)) == 1:

        this_col = np.unique(col_ids)[0]
        # this will be -1 if all are unassigned
        consensus = this_col

    # alternatively, if we have multiple values, do we have only one when we remove -1s/
    else:
        unique_cols = np.unique(col_ids)
        if -1 in unique_cols:

            unique_cols = unique_cols[unique_cols != -1]
        # if we now only have one value!
        if len(unique_cols) == 1:

            consensus = unique_cols[0]

        # if we still have multiple here
        else:

            # get counts for each unique column
            totals = np.array(
                [sum(syn_counts[np.where(col_ids == u_col)]) for u_col in unique_cols]
            )

            # check if there is a tie
            # Find the two highest values efficiently
            largest_two = np.partition(totals, -2)[-2:]
            # Check if they are the same
            if largest_two[0] == largest_two[1]:
                consensus = -2
                warn(
                    f"No consensus was found for column {col}, giving it an assignment of -2"
                )
            else:
                consensus = unique_cols[np.where(totals == totals.max())][0]
    return consensus


def make_consensus_ids(cols: Columns) -> np.ndarray:
    """
    Compute consensus column IDs for all columns based on synapse counts.

    This function determines a consensus column ID for each column by resolving
    conflicts among multiple assigned column IDs using the following rules:

    1. If all assigned IDs for a column are the same, that value is used.
    2. If multiple values exist but removing unassigned (-1) values leaves only one,
       that remaining value is used.
    3. If multiple values remain, the consensus is determined by summing synapse counts
       for each unique column ID and selecting the ID with the highest total.
    4. If there is a tie for the highest synapse count, the function assigns -2 to
       indicate no clear consensus.

    Parameters
    ----------
    cols : NeuOptics.Columns
        An object containing column assignment data, including `Assigned_columns`
        and `Synapse_counts`.

    Returns
    -------
    np.ndarray
        An array of consensus column IDs computed for all columns.
    """
    return np.array(
        [
            _make_consensus(c, cols.Assigned_columns[c], cols.Synapse_counts[c])
            for c in cols.Column_ids
        ]
    )


def _synapse_counts_row(cols,i,m):
    # create an empty array of length = data columns
    row_data = np.zeros(m+)
    for j in range(len(cols.Assigned_columns[i])):

        # if we assigned -1 here
        if cols.Assigned_columns[i][j] == -1:
            pass
        else:
            # column index is column id - 1
            ind = cols.Assigned_columns[i][j]-1
            row_data[ind] += cols.Synapse_counts[i][j]
    return row_data

def synapse_count_matrix(cols,assignment_df):

    n = len(cols.Column_ids)
    m = len(assignment_df)
    mat = np.zeros((n,m))

    for i in range(n):
        mat[i] = _synapse_counts_row(cols,i,m)
    
    return mat

def hungarian_tie_handling(prob_matrix):
    """
    Finds the optimal set of [row, column] indices that maximize the sum of assigned values
    while handling ties systematically:
    - Rows with all zeros are ignored (assigned np.nan).
    - Non-zero ties are resolved by testing all possibilities for global optimization.
    - Ensures that no column is assigned to more than one row.

    Parameters:
        prob_matrix (numpy.ndarray): An n x m array of probabilities (values between 0 and 1).

    Returns:
        list: Optimal assignments (row, column) with np.nan for rows that are ignored.
    """
    n, m = prob_matrix.shape
    assignments = [None] * n  # Initialize assignments
    used_columns = set()  # Track assigned columns

    for i, row in enumerate(prob_matrix):
        max_val = np.max(row)
        tied_indices = np.where(row == max_val)[0]

        if max_val == 0:
            # Case 1: All values in the row are zero
            assignments[i] = np.nan
            continue

        # Track all valid columns for this row
        valid_columns = [col for col in tied_indices if col not in used_columns]
        if not valid_columns:
            # No valid column to assign
            assignments[i] = np.nan
            continue

        # Test all valid columns and pick the best option
        best_score = -np.inf
        best_assignment = None

        for col in valid_columns:
            # Create a temporary matrix where this row assigns to `col`
            temp_matrix = prob_matrix.copy()
            temp_matrix[i, :] = 0  # Zero out the row
            temp_matrix[i, col] = max_val  # Assign the tied value

            # Solve the assignment problem for the rest
            cost_matrix = -temp_matrix  # Negate for maximization
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            current_score = -cost_matrix[row_indices, col_indices].sum()

            if current_score > best_score:
                best_score = current_score
                best_assignment = col

        # Assign the best column, if found
        if best_assignment is not None:
            assignments[i] = (i, best_assignment)
            used_columns.add(best_assignment)
        else:
            # Fallback: assign the first available valid column
            fallback_column = valid_columns[0]
            assignments[i] = (i, fallback_column)
            used_columns.add(fallback_column)

    return assignments

def make_consensus_hungarian(cols, assignment_df, return_dict = True):

    mat = synapse_count_matrix(cols,assignment_df)
    assignment = hungarian_tie_handling(mat)
    if return_dict:
        dict_to_return = dict()
        for i in cols.Column_ids:
            assignment_tuple = assignment[i]
            try:
                dict_to_return[i] = assignment[i][1]
            except:
                dict_to_return[i] = np.nan
        return dict_to_return
    else:
        return  assignment