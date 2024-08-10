By: Karthik Muthukumar

This project is done to model the Seasonally Adjusted HPI growth of homes in the Pacific division in the United States of America.

HPI (House Price Index) is the measure of changes in prices of a house overtime and can inturn be used as a good indicator of the state of the economy (whether there is inflation or not).

I hope this can be improved and used to identify future trends.

This is done through using Scikit-Learn to create a Polynomial Regression model to plot data from trends we see in the past and its intent is to approximate future HPI values for a specific month (over the next three years up to 2027) given the trends in the past assuming economic stability. For my dataset I am using seasonally adjusted data to eliminate variations because of season and to see more raw trends. I also do feature scaling and a train test cross validation split to try to make it as accurate as possible. The user is also prompted to predict a month ranging from 1 to 438 corresponding from 1971 to mid 2027.

Make sure to pip install the required packages such as matplotlib, scikit-learn, numpy, and pandas.

Data: https://www.fhfa.gov/data/hpi/datasets?tab=monthly-data

STEPS:
1. Download all files and open the folder through an IDE
2. Just run "regression model home prices"


