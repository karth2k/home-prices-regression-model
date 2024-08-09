This project is done to model the Seasonally Adjusted HPI growth of homes in the Pacific division in the United States of America.

HPI (House Price Index) is the measure of changes in prices of a house overtime and can inturn be used as a good indicator of the economy (whether there is inflation or not).

I hope this can be improved and used to identify future trends.

This is done through using a Polynomial Regression model to plot data from trends we see in the past and its intent is to approximate future HPI values for a specific month given the trends in the past assuming economic stability. For my dataset I am using seasonally adjusted data to eliminate variations because of season and to see more raw trends. I also do feature scaling and a train test cross validation split to try to make it as accurate as possible.

I will soon try to implement a Ridge Regression as my cross validation mean squared error seems to be a bit high

Data: https://www.fhfa.gov/data/hpi/datasets?tab=monthly-data

This is still a work in progress...


