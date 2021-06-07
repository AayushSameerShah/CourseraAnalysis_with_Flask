import pandas as pd
import numpy as np

df = pd.read_pickle('../data/scraped_data')

def fetch_list():
    unis = list(np.sort(df.university.unique()))
    type_ = list(np.sort(df.type.unique()))
    diff = list(np.sort(df.difficulty.dropna().unique()))
    return (unis, type_, diff)
