def assign_preallocated_columns(cols:NeuOptics.Columns,data:pd.DataFrame | None = None, bind = True) -> None | dict:
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
        data = NeuOptics.flywire_column_assignment_table()
    
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
        col_dict[col_id] = assigned_columns

    if bind:
        cols.Assigned_columns = col_dict
    else:
        return col_dict