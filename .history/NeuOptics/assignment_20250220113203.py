from pandas import DataFrame
from .columns import Columns
from .flywire import flywire_column_assignment_table
from numpy import array, ndarray, unique

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
        col_dict[col_id] = array(assigned_columns)

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


def make_consensus_ids(cols: NeuOptics.Columns) -> np.ndarray:
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
            _make_consensus(c, T4_cols.Assigned_columns[c], T4_cols.Synapse_counts[c])
            for c in T4_cols.Column_ids
        ]
    )