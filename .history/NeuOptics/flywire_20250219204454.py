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



@jit
def _fwy_col_transform(coords: jnp.ndarray) -> jnp.ndarray:
    # get q and r
    hex_coords = jnp.column_stack((coords[:,0] - coords[:,1], -coords[:,0]))
    # define s
    hex_coords = jnp.column_stack((hex_coords,-hex_coords[:,0] - hex_coords[:,1]))
    return hex_coords