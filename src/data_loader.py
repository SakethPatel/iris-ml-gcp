import pandas as pd
#comment
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
