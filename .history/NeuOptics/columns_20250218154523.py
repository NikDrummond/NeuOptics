import pandas as pd

def flywire_column_assignment_table() -> pd.Dataframe:
    return pd.read_csv('/home/nik/Desktop/Optic_Lobe_data/Jan_final/column_assignment.csv')