import vedo as vd
from typing import List
from Neurosetta import Forest_graph
import numpy as np

def _fit_sphere(coords: np.ndarray)  :
    return vd.fit_sphere(coords)

def fit_sphere(N: List | Forest_graph):

    # if we are given a list of neurons
    if isinstance(N, List): 
        pass

    # else if we are given a Forest_graph
    elif isinstance(N, Forest_graph):
        pass

    else:
        raise TypeError('N_all must be a list of neurosetta neurons, or a Neurosetta Forest_graph')