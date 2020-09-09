import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# house prices file
home_price = pd.read_csv("C:/Users/z011348/Desktop/ML/input/homeprices.csv")
# print(home_price)
price = pd.read_csv("C:/Users/z011348/Desktop/ML/input/prices.csv")

X = home_price.drop("price", axis=1)
y = home_price["price"]

np.random.seed(42)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2)

model = LinearRegression()
model.fit(X_train,y_train)

y_preds = model.predict(X_test)
# print(y_preds)
print("Model score accuracy:")
print(model.score(X_train, y_train))

# print(model.coef_)
# print(model.intercept_)

pred_prices = model.predict(price)
# print(pred_prices)
price["prediction_price"] = pred_prices
price.to_csv("C:/Users/z011348/Desktop/ML/output/prediction_prices.csv", index=False)
