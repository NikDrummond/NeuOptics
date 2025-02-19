import pandas as pd
from pathlib import Path

def flywire_column_assignment_table() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parent.parent / "Data" / "data_file.csv"
    return pd.read_csv('./data/column_assignment.csv')


