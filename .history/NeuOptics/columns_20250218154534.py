import pandas as pd

def flywire_column_assignment_table() -> pd.Dataframe:
    return pd.read_csv('/data/column_assignment.csv')