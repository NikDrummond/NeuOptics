import pandas as pd
from pathlib

def flywire_column_assignment_table() -> pd.DataFrame:
    return pd.read_csv('./data/column_assignment.csv')


