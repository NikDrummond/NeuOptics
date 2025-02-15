import vedo as vd
from typing import List
from Neurosetta import Forest_graph
import numpy as np
import GeoJax as gj


def _fit_sphere(coords: np.ndarray)  :
    return vd.fit_sphere(coords)

def fit_sphere(N: List | Forest_graph):
    """ From a list of neurons, or a neurosetta Forest_graph fit a sphere to all coordinates in the dataset"""

    # if we are given a list of neurons
    if isinstance(N, List): 
        coords = np.vstack([n.graph.vp['coordinates'].get_2d_array().T for n in N])

    # else if we are given a Forest_graph - currently does not take synapses into account
    elif isinstance(N, Forest_graph):
        coords = N.all_coords()

    else:
        raise TypeError('N_all must be a list of Neurosetta neurons, or a Neurosetta Forest_graph')
    return _fit_sphere(coords)