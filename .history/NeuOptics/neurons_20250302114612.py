from pandas import DataFrame
from numpy import where, isnan
from Neurosetta import Tree_graph, get_all_centres, get_all_roots

from .columns import Columns
from .pnt_assignment import max_pnt_assignment, vmf_likelihood_matrix, 


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
    

def Neuron_column_assignment(cols: Columns, N_all: nr.Forest_graph, point_type:str = 'root', bind: bool = True) -> List | None:
    """Uses the Hungarian algorithm to uniquely assign neuron points in N_all to specific column in cols.
    cols object MUST have mu and kappa attributes. If this has not been done already, use `NeuOptics.fit_vonMises_Fisher`

    Unassigned Neurons are given an assignment value of -1

    Parameters
    ----------
    cols : Columns
        Column object to assign neuron points to 
    N_all : nr.Forest_graph
        Forest graph of all neurons
    point_type : str, optional
        Must be root or centre. This determines what kind of point from the neuron is assigned to a column. If root,
        this will be the root of the Neuron, eg the soma or first branch point of the dendritic tree. 
        If centre, this will be the centre of mass of point coordinates which make up the neuron graph. 
        By default 'root'
    bind : bool, optional
        If True, a graph property will be bound to each neuron specifying it's column index in the partnered Columns object.
        If point type was root, this will be "root_column", if centre this will be "centre_column".
        Otherwise, return a List of neuron index to Column index.
        By default True


    Returns
    -------
    List | None
        If Bind is False, returns List of tuples of Neuron index to Column index

    Raises
    ------
    AttributeError
        If point_type is neither "root" nor "centre"
    """

    # make sure columns has fitted vmf
    assert (hasattr(cols, 'mu')) & (hasattr(cols, 'kappa')), 'Columns object must have mu and kappa attribute, \n Please fit the vonMises-Fisher using neuOptics.fit_vonMises_Fisher'

    # get coordinates
    if point_type == 'root':
        coords = get_all_roots(N_all)
    elif point_type == 'centre':
        coords = get_all_centres(N_all)
    else:
        raise AttributeError(f'{point_type} is not a valid option, expected root or centre')
    

    # get likelihood matrix
    mat = vmf_likelihood_matrix(cols,coords, norm = True)
    # set nan values to 0
    mat[np.where(np.isnan(mat))] = 0
    # make assignment list
    a_list = NeuOptics.hungarian_tie_handling(mat.T)
    # mark unassigned as -1
    a_list = [(v,-1) if np.isnan(a_list[v]).any() else a_list[v] for v in sub_dict[s].graph.iter_vertices()]
    # bind or return dict
    if bind:
        if point_type == 'root':
            gp_label = 'root_column'
        else:
            gp_label = 'centre_column'
        for v in N_all.graph.iter_vertices():
            assignment = a_list[v]
            N_all.graph.vp['Neurons'][v].graph.gp[gp_label] = N_all.graph.new_gp('int',assignment)
    else:
        return a_list