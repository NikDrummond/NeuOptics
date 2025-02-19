import pandas as pd
from pathlib import path

def flywire_column_assignment_table() -> pd.DataFrame:
    return pd.read_csv('./data/column_assignment.csv')


