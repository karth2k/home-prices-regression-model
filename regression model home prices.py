import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures


data = pd.read_csv("Housing_Price_Index_Pacific_Division_-_Sheet1.csv")


X = data['month'].values
y = data['index_sa'].values


poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(X.reshape(-1, 1))

poly_model = LinearRegression()

poly_model.fit(poly_features, y)
y_pred = poly_model.predict(poly_features)


future_years = np.arange(X.max()+1, X.max()+121).reshape(-1, 1)
future_features = poly.transform(future_years)
future_pred = poly_model.predict(future_features)





plt.scatter(X,y)
plt.plot(X,y_pred, c = 'red')
plt.plot(future_years, future_pred, c='green')
plt.xlabel("Month (starting from January 1991)")
plt.title("Pacific Division Dataset (taken from Federal Housing Finance Agency)")
plt.ylabel("Housing Price Index (Seasonally Adjusted)")
plt.show()

