# Implementation of Simple Linear Regression using Scikit-learn
# Create a model, train, test, and use the model
# Data: FuelConsumptionCo2.csv
# Find the relationship between Co2 Emissions and other independent features

# Todo: Importing required packages
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np


# Todo: Download Data
# (already downloaded)

# Todo: Understanding Data:
# FuelConsumptionCo2.csv, which contains model specific fuel consumption ratings and
# estimated carbon dioxide emissions for new light-duty vehicles for retail sale in Canada.

# MODEL YEAR e.g. 2014
# MAKE e.g. Acura
# MODEL e.g. ILX
# VEHICLE CLASS e.g. SUV
# ENGINE SIZE e.g. 4.7
# CYLINDERS e.g 6
# TRANSMISSION e.g. A6
# FUEL CONSUMPTION in CITY(L/100 km) e.g. 9.9
# FUEL CONSUMPTION in HWY (L/100 km) e.g. 8.9
# FUEL CONSUMPTION COMB (L/100 km) e.g. 9.2
# CO2 EMISSIONS (g/km) e.g. 182 –> low –> 0


# Todo: Reading the data in
df = pd.read_csv("../data/FuelConsumption.csv")

# Check whether data is loaded into dataframe or not
# print(df.head())

# get data info, shapes (rows, columns), types
# print(df.shape)
# print(df.info)
# print(df.dtypes)


# Todo: Data Exploration

# 1. Summarize the data statistics
data_statistics = df.describe()
# print(data_statistics)

# Save summarized data statistics into csv
# data_statistics.to_csv("../results/regression/FuelConsumption_data_statistics.csv", sep=',')


# Todo: Select some features to explore more
# Features taken: 'ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS'
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
# print(cdf.head(10))


# Todo: Plot each of these features
viz = cdf[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
# viz.hist()
# plt.savefig("../results/regression/cdf_FuelConsumption.png")
# plt.show()


# Todo: Plot each of these features vs the Emission, to see how linear is their relation
# 1. 'FUELCONSUMPTION_COMB' vs 'CO2EMISSIONS'
# plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS)
# plt.title("Fuel Consumption_COMB vs CO2 Emission")
# plt.xlabel("Fuel Consumption COMB")
# plt.ylabel("CO2 Emission")
# plt.savefig("../results/regression/Fuel_Consumption_COMB_vs_CO2_Emission.png")
# plt.show()

# 2. 'ENGINESIZE' vs 'CO2EMISSIONS'
# plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS)
# plt.title("Engine Size vs CO2 Emission")
# plt.xlabel("Engine Sise")
# plt.ylabel("CO2 Emission")
# plt.savefig("../results/regression/Engine_Size_vs_CO2_Emission.png")
# plt.show()

# 3. 'CYLINDERS' vs 'CO2EMISSIONS'
# plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS)
# plt.title("Cylinder Size vs CO2 Emission")
# plt.xlabel("Cylinder Size")
# plt.ylabel("CO2 Emission")
# plt.savefig("../results/regression/Cylinder_Size_vs_CO2_Emission.png")
# plt.show()


# Todo: Creating Train and Test dataset
# Train/Test Split involves splitting the dataset into training and testing sets respectively, which are mutually exclusive.
# Train with training set and test with testing set
# It will provide more accurate evaluation on out-of-sample accuracy because testing dataset is not part of training dataset
# Split dataset into trains and test sets, 80% of the entire data for training, and the 20% for testing

# Create a mask to select random rows using np.random.rand() function
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]
# print(train.head())
# print(test.head())

# Todo: Simple Regression model
# Linear Regression fits a linear model with coefficients B=(B1,..,Bn) to minimize the 'residual sum of squares'
# between the actual value y in the dataset, and the predicted value y_hat using linear approximation.

# Train data distribution
# plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS)
# plt.xlabel("Engine Size")
# plt.ylabel("CO2 Emission")
# plt.show()


# Todo: Model data using sklearn package
from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
# print(train_y)
regr.fit(train_x, train_y)

# Calculate coefficient and intercept
# Coefficient and intercept in the simple linear regression, are the parameters of the fit line
print("Coefficients: {}".format(regr.coef_))
print("Intercept: {}".format(regr.intercept_))

# Todo: Plot the fit line
# plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS)
# plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
# plt.title("Engine Size vs CO2 Emission")
# plt.xlabel("Engine Size")
# plt.ylabel("CO2 Emission")
# plt.savefig("../results/regression/Engine_Size_vs_CO2_Emission_SLR.png")
# plt.show()

# Todo: Evaluation
# Compare the actual values and predicted values to calculate the accuracy of a regression model
# Evaluation metrics provide a key role in the development of a model, as it provides insight to
# areas that require improvement.
# There are different model evaluation metrics, lets use MSE here to calculate the accuracy of our
# model based on the test set

# Mean absolute error: It is the mean of the absolute value of the errors - the easiest one
# Mean Squared Error (MSE): Mean Squared Error (MSE) is the mean of the squared error
# Root Mean Squared Error (RMSE)
# R-squared is not error, but is a popular metric for accuracy of your model. It represents how
# accurate the model is.

from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_predicted = regr.predict(test_x)


# Calculate the metrics
mean_abs_error = np.mean(np.absolute(test_y_predicted - test_y))
mean_sqr_error = np.mean((test_y_predicted - test_y) ** 2)
r2 = r2_score(test_y, test_y_predicted)

print("Mean Absolute Error: {:.2f}".format(mean_abs_error))
print("Residual sum of squares (MSE): {:.2f}".format(mean_sqr_error))
print("R2-score: {:.2f}".format(r2))

# Output:
# Coefficients: [[39.10966552]]
# Intercept: [124.41920537]
# Mean Absolute Error: 23.17
# Residual sum of squares (MSE): 884.10
# R2-score: 0.75


# Todo: Conclusion:
# Usually, the larger the R2, the better the regression model fits your observations
# Resource for R-squared
# https://statisticsbyjim.com/regression/interpret-r-squared-regression/
