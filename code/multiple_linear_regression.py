# Implementation of Multiple Linear Regression using Scikit-learn
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
# Features taken: 'ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS'
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
# print(cdf.head(10))


# Todo: Plot each of these features vs the Emission, to see how linear is their relation
# 2. Plot Emission values with respect to Engine size: 'ENGINESIZE' vs 'CO2EMISSIONS'
# plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS)
# plt.title("Engine Size vs CO2 Emission")
# plt.xlabel("Engine Sise")
# plt.ylabel("CO2 Emission")
# plt.savefig("../results/regression/Engine_Size_vs_CO2_Emission.png")
# plt.show()


# 2. 'FUELCONSUMPTION_COMB' vs 'CO2EMISSIONS'
# plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS)
# plt.title("Fuel Consumption_COMB vs CO2 Emission")
# plt.xlabel("Fuel Consumption COMB")
# plt.ylabel("CO2 Emission")
# plt.savefig("../results/regression/Fuel_Consumption_COMB_vs_CO2_Emission.png")
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

# Todo: Multiple Linear Regression model
# Linear Regression fits a linear model with coefficients B=(B1,..,Bn) to minimize the 'residual sum of squares'
# between the actual value y in the dataset, and the predicted value y_hat using linear approximation.

# Train data distribution
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS)
plt.xlabel("Engine Size")
plt.ylabel("CO2 Emission")
plt.show()

# Todo: Multiple Regression Model - using sklearn package
from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

regr.fit(train_x, train_y)

# Calculate coefficient
print("Coefficients: {}".format(regr.coef_))

# Note:
# As mentioned before, Coefficient and Intercept , are the parameters of the fit line.
# Given that it is a multiple linear regression, with 3 parameters, and knowing that the parameters are
# the intercept and coefficients of hyperplane, sklearn can estimate them from our data.
# Scikit-learn uses plain Ordinary Least Squares method to solve this problem.

# Ordinary Least Squares (OLS)
# ----------------------------
# OLS is a method for estimating the unknown parameters in a linear regression model.
# OLS chooses the parameters of a linear function of a set of explanatory variables by minimizing
# the sum of the squares of the differences between the target dependent variable and those predicted
# by the linear function. In other words, it tries to minimizes the sum of squared errors (SSE) or
# mean squared error (MSE) between the target variable (y) and our predicted output ( 𝑦̂  ) over all
# samples in the dataset.
# OLS can find the best parameters using of the following methods:
# - Solving the model parameters analytically using closed-form equations
# - Using an optimization algorithm (Gradient Descent, Stochastic Gradient Descent, Newton’s Method, etc.)


# Todo: Prediction
test_x = np.asanyarray(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
y_hat = regr.predict(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])


# Calculate the metrics
mean_sqr_error = np.mean((y_hat - test_y) ** 2)
print("Residual sum of squares (MSE): {:.2f}".format(mean_sqr_error))

# Explained variance score: 1 is perfect prediction
print('Variance score: {:.2f}'.format(regr.score(test_x, test_y)))

# Output:
# Coefficients: [[10.97819789  7.7010866   9.49958338]]
# Residual sum of squares (MSE): 547.60
# Variance score: 0.85


# Explained variance regression score:
# If  𝑦̂   is the estimated target output, y the corresponding (correct) target output, and Var is Variance,
# the square of the standard deviation, then the explained variance is estimated as follow:
# 𝚎𝚡𝚙𝚕𝚊𝚒𝚗𝚎𝚍𝚅𝚊𝚛𝚒𝚊𝚗𝚌𝚎(𝑦,𝑦̂ )=1−((𝑉𝑎𝑟𝑦−𝑦̂  )/𝑉𝑎𝑟𝑦)
# The best possible score is 1.0, lower values are worse.
