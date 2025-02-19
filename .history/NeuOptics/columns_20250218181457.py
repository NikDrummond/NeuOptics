import pandas as pd
import importlib.resources as pkg_resources
import NeuOptics  # Ensure this is the correct package name

def flywire_column_assignment_table():
    """Load the column assignment data file from the installed package."""
    try:
        with pkg_resources.files(NeuOptics).joinpath("data/flywire_column_coordinates.csv").open("r") as f:
            return pd.read_csv(f)
    except FileNotFoundError:
        raise FileNotFoundError("Data file 'column_assignment.csv' not found in the installed package. "
                                "Ensure NeuOptics is installed correctly and includes the 'data' folder.")
