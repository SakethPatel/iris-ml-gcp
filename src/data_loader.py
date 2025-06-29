import pandas as pd
#commen_t
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
