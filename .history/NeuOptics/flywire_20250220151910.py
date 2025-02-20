import pandas as pd
import jax.numpy as jnp
from jax import jit
import importlib.resources as pkg_resources
import NeuOptics  # Ensure this is the correct package name

def flywire_column_coordinate_table():
    """Load the column coordinates data file from the installed package."""
    try:
        with pkg_resources.files(NeuOptics).joinpath("data/flywire_column_coordinates.csv").open("r") as f:
            return pd.read_csv(f)
    except FileNotFoundError:
        raise FileNotFoundError("Data file 'column_assignment.csv' not found in the installed package. "
                                "Ensure NeuOptics is installed correctly and includes the 'data' folder.")

def flywire_column_assignment_table():
    """Load the column coordinates data file from the installed package."""
    try:
        with pkg_resources.files(NeuOptics).joinpath("data/N_id_column_assignment.csv").open("r") as f:
            return pd.read_csv(f)
    except FileNotFoundError:
        raise FileNotFoundError("Data file 'N_id_column_assignment.csv' not found in the installed package. "
                                "Ensure NeuOptics is installed correctly and includes the 'data' folder.")


@jit
def _fwy_col_transform(coords: jnp.ndarray) -> jnp.ndarray:
    # get q and r
    hex_coords = jnp.column_stack((coords[:,0] - coords[:,1], -coords[:,0]))
    # define s
    hex_coords = jnp.column_stack((hex_coords,-hex_coords[:,0] - hex_coords[:,1]))
    return hex_coords

@jit
def _fwy_coord_transform(arr: jnp.ndarray) -> jnp.ndarray:
    q = arr[0] - arr[1]
    r = -arr[0]
    s = -q-r
    return jnp.array([q,r,s])

def pq_to_qrs_transform(coords: jnp.ndarray) -> jnp.ndarray:
    """Flywire uses a [p,q] coordinate system, where as HexCraft uses a 
    [q,r,s] coordinate system. This function allows for transformations from
    [p,q] to [q,r,s]. here: q = p - q; r = -p, and s = -q-r 

    Parameters
    ----------
    coords : jnp.ndarray
        Array of [p,q] coordinates

    Returns
    -------
    jnp.ndarray
        _description_

    Raises
    ------
    AttributeError
        _description_
    """
    if coords.ndim == 1:
        return _fwy_coord_transform(coords)
    elif coords.ndim == 2:
        return _fwy_col_transform(coords)
    else:
        raise AttributeError('Input coords must be a single pq point, or array of them.')
    