import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression

def predict(model, scaler, poly):
    i = 0
    while i<1:
        try:
            month = int(input('Predict a month of your choice (Must be a number between 1 (being 1971) and 438 (end of spring 2027)): '))
            if month <=0 or month>438:
                raise TypeError()
            month_scaled = scaler.transform(np.array([[month]]))
            month_poly = poly.transform(month_scaled)

            hpi_pred = model.predict(month_poly)
            print(f"Predicted HPI value for month number {month} is {hpi_pred}.")
            i=i+1
        except ValueError:
            print(f"Invalid input try again. Make sure it input a number.")
        except TypeError:
            print("Make sure the number is inbetween 1 and 438 inclusive.")