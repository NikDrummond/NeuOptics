from pandas import DataFrame
from Neurosetta import Tree_graph

from .columns import Columns
from .pnt_assignment import max_pnt_assignment


def N_assign_input_synapse_columns(N: Tree_graph, cols: Columns, bind = True) -> DataFrame | None:
    """Maximum Likeleihood assignment of input synapses in N to columns in cols

    Parameters
    ----------
    N : Neurosetta.Tree_graph
        Tree graph Neuron representation to get synapse inputs from
    cols : NeuOptics.Columns
        Column object to map synapse coordinates to
    bind : bool, optional
        If True, 'Assigned_column' column will be added to N.graph.gp['inputs].
        Otherwise, a copy of N.graph.gp['inputs'] is returned with the added column. By default True

    Returns
    -------
    None | pd.Dataframe
        If bind is False input synapse data frame is returned
    """

    input_df = N.graph.gp['inputs']
    syn_coords = input_df[['graph_x','graph_y','graph_z']].values
    col_assignments = max_pnt_assignment(cols, syn_coords)
    if bind:
        N.graph.gp['inputs']['Assigned_column'] = col_assignments
    else:
        df = input_df.copy()
        df['Assigned_column'] = col_assignments
        return df