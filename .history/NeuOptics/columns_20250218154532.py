import pandas as pd

def flywire_column_assignment_table() -> pd.Dataframe:
    return pd.read_csv('//column_assignment.csv')