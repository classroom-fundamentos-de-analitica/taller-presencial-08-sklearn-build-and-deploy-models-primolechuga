import pickle

import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("house_data.csv", sep=",")

features = df[
    [
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "waterfront",
        "condition",
    ]
]

target = df[["price"]]

estimator = LinearRegression()
estimator.fit(features, target)

print(estimator.coef_)
print(estimator.intercept_)
with open("house_predictor.pickle", "wb") as file:
    pickle.dump(estimator, file)
    
    
    
    
    
    
    
    
    
    
    