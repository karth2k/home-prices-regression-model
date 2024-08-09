import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split



data = pd.read_csv("Housing_Price_Index_Pacific_Division_-_Sheet1.csv")                    #Reads the data from the csv file I downloaded from the website


X = data['month'].values.reshape(-1, 1)                                                    #Input Feature is the month the reshape changes the 1D array to a 2D array for Scikit
y = data['index_sa'].values                                                                #Output Targer is the HPI value during the month

X_train, X_, y_train, y_ = train_test_split(X, y, test_size = .4, random_state = 5)        #Splits the data into the training set which is a random 60% of the data, cross validation and test set are both 20% of the data set
X_cv, X_test, y_cv, y_test = train_test_split(X_, y_, test_size = .5, random_state =5)
del X_, y_  #Removes the temporary values


scaler_poly = StandardScaler()                                                             #Implements feature scaling
X_train_scaled = scaler_poly.fit_transform(X_train)
X_cv_scaled = scaler_poly.transform(X_cv)
X_test_scaled = scaler_poly.transform(X_test)

print(f"The computed mean of the training set is: {scaler_poly.mean_.squeeze()}")
print(f"The computed standard deviation of the training set is {scaler_poly.scale_.squeeze()}\n")



poly = PolynomialFeatures(degree=2, include_bias=False)                                    #Form the polynomial features with the specific degree I want
X_train_poly = poly.fit_transform(X_train_scaled)
X_cv_poly = poly.transform(X_cv_scaled)
X_test_poly = poly.transform(X_test_scaled)

poly_model = LinearRegression()                                                            #These lines train the model on the given transformed training data 
poly_model.fit(X_train_poly, y_train)

yhat_train = poly_model.predict(X_train_poly)                                              #Feeds the scaled training data and feeds it to the model checks how well model fits training data
train_mse = mean_squared_error(y_train, yhat_train)/2
print(f"Training Mean Squared Error: {train_mse}")

yhat_cv = poly_model.predict(X_cv_poly)                                                    #Checks how well model fits unseen data
cv_mse = mean_squared_error(y_cv, yhat_cv)
print(f"Cross Validation Mean Squared Error: {cv_mse}")

yhat_test = poly_model.predict(X_test_poly)                                               
test_mse = mean_squared_error(y_test, yhat_test)
print(f"Test Mean Squared Error: {test_mse}")




y_pred = poly_model.predict(poly.transform(scaler_poly.transform(X)))


future_years = np.arange(X.max()+1, X.max()+61).reshape(-1, 1)                           #Predicts the future HPI values for the next 5 years
future_features = poly.transform(scaler_poly.transform(future_years))
future_pred = poly_model.predict(future_features)


plt.scatter(X,y)                                                                         #Graphs the plot
plt.plot(X,y_pred, c = 'red') 
plt.plot(future_years, future_pred, c='green')
plt.xlabel("Month (starting from January 1991)")
plt.title("Pacific Division Dataset (taken from Federal Housing Finance Agency)")
plt.ylabel("Housing Price Index (Seasonally Adjusted)")
plt.show()




