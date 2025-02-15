import vedo as vd
from typing import List
from Neurosetta import Forest_graph

def _fit_sphere(coords:):
    return vd.fit_sphere(coords)

def fit_sphere(N:):
    if isinstance(N, List)