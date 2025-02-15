import vedo as vd
from typing import List
import Neurosetta as nr
import numpy as np
import GeoJax as gj
from typing import Union, List


def _fit_sphere(coords: np.ndarray)  :
    return vd.fit_sphere(coords)

def fit_sphere(N: List | nr.Forest_graph):
    """ From a list of neurons, or a neurosetta Forest_graph fit a sphere to all coordinates in the dataset
    
    Currently, this does not take input and output synapses into account!
    """

    # if we are given a list of neurons
    if isinstance(N, List): 
        coords = np.vstack([n.graph.vp['coordinates'].get_2d_array().T for n in N])

    # else if we are given a Forest_graph - currently does not take synapses into account
    elif isinstance(N, nr.Forest_graph):
        coords = N.all_coords()

    else:
        raise TypeError('N_all must be a list of Neurosetta neurons, or a Neurosetta Forest_graph')
    return _fit_sphere(coords)

def _N_sphere_projection(N: Union[np.ndarray, nr.Tree_graph], sphere: Union[vd.Sphere, None] = None, r: Union[float, None] = None, c:Union[np.ndarray, None] = None, bind: bool = False) -> np.ndarray | None:

    ### do we already have coordinates or do we need to get them?
    if isinstance(N, nr.Tree_graph):
        coords = nr.g_vert_coords(N)
    elif isinstance(N, np.ndarray):
        coords = N
    else:
        raise TypeError('N must be a np.ndarray or neurosetta.Tree_graph')
    
    ### Figure out how we are using sphere, r and c
    # make sure all of sphere, r, and s are not none
    if (sphere is None) & (r is None) & (c is None):
        raise AttributeError('All of sphere, r, and c cannot be None')

    # check if sphere is not None and use it
    if sphere != None:
        # make sure sphere is a vd.Sphere, and if it is get radius and center of mass
        assert isinstance(sphere, vd.Sphere), 'If sphere is provided, if must be a vedo.Sphere object'
        r = sphere.radius
        c = sphere.center_of_mass()
    else:
        # make sure that both r and c have been provided properly
        assert isinstance(r, float), 'Given radius (r) must be a float'
        assert isinstance(c, np.ndarray), 'Given center of mass (c) must be np.ndarray'
        assert len(c) == coords.shape[1], ' given center of mass must have same dimension as given coordinates'
    
    # project coordinates
    s_coords = gj.project_to_sphere(coords,r,c)

    # tracking later so we know what to return
    inputs = False
    outputs = False

    # do we have input synapses to handle?
    if nr.g_has_property(N, 'inputs','g'):
        inputs = True
        # coordinates of inputs not mapped to graph
        coords_in_not_mapped = gj.project_to_sphere(N.graph.gp['inputs'][['post_x','post_y','post_z']].values,r,c)
        # coordinates of inputs mapped to graph
        coords_in_mapped = gj.project_to_sphere(N.graph.gp['inputs'][['graph_x','graph_y','graph_z']].values,r,c)
    
    # do we have output synapses to handle?
    if nr.g_has_property(N, 'outputs', 'g'):
        outputs = True
        # coordinates of outputs not mapped to graph
        coords_out_not_mapped = gj.project_to_sphere(N.graph.gp['outputs'][['pre_x','pre_y','pre_z']].values,r,c)
        # coordinates of outputs mapped to graph
        coords_out_mapped = gj.project_to_sphere(N.graph.gp['outputs'][['graph_x','graph_y','graph_z']].values,r,c)

    # handling binding
    if (isinstance(N, nr.Tree_graph)) and (bind):
        N.graph.vp['coordinates'].set_2d_array(s_coords.T)
        N.graph.gp['inputs'][['post_x','post_y','post_z']] = coords_in_not_mapped
        N.graph.gp['inputs'][['graph_x','graph_y','graph_z']] = coords_in_mapped
        N.graph.gp['outputs'][['pre_x','pre_y','pre_z']] = coords_out_not_mapped
        N.graph.gp['outputs'][['graph_x','graph_y','graph_z']] = coords_out_mapped
        
    else:
        if inputs & outputs:
            return s_coords, coords_in_not_mapped, coords_in_mapped, coords_out_not_mapped, coords_out_mapped
        elif inputs:
            return s_coords, coords_in_not_mapped, coords_in_mapped
        elif outputs:
            return s_coords, coords_out_not_mapped, coords_out_mapped
        else:
            return s_coords

def project_to_sphere(
    N: Union[List, nr.Tree_graph, nr.Forest_graph],
    sphere: Union[vd.Sphere, None] = None,
    r: Union[float, None] = None,
    c: Union[np.ndarray, None] = None,
    bind: bool = False,
) -> np.ndarray | None:
    """
    
    """


    # unpack and figure out how we are handling N

    # if this is a list, iterate through it's values
    if isinstance(N, List):
        for n in N:
            return [NeuOptics._N_sphere_projection(N, sphere, r, c, bind)]

    # if this is a single tree graph
    elif isinstance(N, nr.Tree_graph):
        return NeuOptics._N_sphere_projection(N, sphere, r, c, bind)

    # if this is a Forest graph object
    elif isinstance(N, nr.Forest_graph):
        if ~bind:
            data = []
        for v in N.graph.iter_vertices():
            n = N.graph.vp["Neurons"][v]
            if ~bind:
                data.append(NeuOptics._N_sphere_projection(n, sphere, r, c, bind))
            else:
                NeuOptics._N_sphere.projection(n, sphere, r, c, bind)