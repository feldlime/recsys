import joblib
import pandas as pd
import userknn

interactions = pd.read_csv('/Users/dmitry/PycharmProjects/recsys/data/datasets/interactions_processed.csv')
model = joblib.load('/Users/dmitry/PycharmProjects/recsys/data/models/userknn_tined.joblib')

x = model.predict(704055, 10)
