import pandas as pd
import numpy as np
def create_data():
    df = pd.read_csv("housing.csv")
    df2 = df.iloc[:, :-1]
    features = df2.iloc[:, :-1]
    right_answers = df2.iloc[:, -1]
    return features, right_answers