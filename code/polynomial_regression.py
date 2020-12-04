# Implementation of Polynomial Regression using Scikit-learn
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

# Polynomial Regression
# =========================================================================================================================
# Sometimes, the trend of data is not really linear, and looks curvy. In this case we can use Polynomial regression methods.
# In fact, many different regressions exist that can be used to fit whatever the dataset looks like, such as quadratic, cubic,
# and so on, and it can go on and on to infinite degrees.
# In essence, we can call all of these, polynomial regression, where the relationship between the independent variable x and
# the dependent variable y is modeled as an nth degree polynomial in x. Lets say you want to have a polynomial regression
# (let's make 2 degree polynomial): 𝑦=𝑏+𝜃1𝑥+𝜃2𝑥2
# Now, the question is: how we can fit our data on this equation while we have only x values, such as Engine Size?
# Well, we can create a few additional features: 1,  𝑥 , and  𝑥2 .
# PolynomialFeatures() function in Scikit-learn library, drives a new feature sets from the original feature set.
# That is, a matrix will be generated consisting of all polynomial combinations of the features with degree
# less than or equal to the specified degree.
# For example, lets say the original feature set has only one feature, ENGINESIZE.
# Now, if we select the degree of the polynomial to be 2, then it generates 3 features, degree=0, degree=1 and degree=2:

# Todo Polynomial Regression - using sklearn package
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])

poly = PolynomialFeatures(degree=2)
train_x_poly = poly.fit_transform(train_x)
# print(train_x_poly)

# Output:
# [[ 1.    2.4   5.76]
#  [ 1.    1.5   2.25]
#  [ 1.    3.5  12.25]
#  ...
#  [ 1.    3.2  10.24]
#  [ 1.    3.    9.  ]
#  [ 1.    3.2  10.24]]


# fit_transform takes our x values, and output a list of our data raised from power of 0 to power of 2
# (since we set the degree of our polynomial to 2).
# Polynomial regression is a special case of linear regression, with the main idea of how do you select your features.
# Just consider replacing the  𝑥  with  𝑥1 ,  𝑥21  with  𝑥2 , and so on. Then the degree 2 equation would be turn into: 𝑦=𝑏+𝜃1𝑥1+𝜃2𝑥2
# Now, we can deal with it as 'linear regression' problem.
# Therefore, this polynomial regression is considered to be a special case of traditional multiple linear regression.
# So, you can use the same mechanism as linear regression to solve such a problems.
# so we can use LinearRegression() function to solve it:

# Todo: Linear Regression model - using LinearRegression()
clf = linear_model.LinearRegression()
train_y = clf.fit(train_x_poly, train_y)

# Calculate coefficients and intercept
print("Coefficients: {}".format(clf.coef_))
print("Intercept: {}".format(clf.intercept_))

# Output:
# Coefficients: [[ 0.   48.99089231 -1.35645219]]
# Intercept: [109.75978292]

# As mentioned before, Coefficient and Intercept , are the parameters of the fit curvy line.
# Given that it is a typical multiple linear regression, with 3 parameters, and knowing that the
# parameters are the intercept and coefficients of hyperplane, sklearn has estimated them
# from our new set of feature sets.
# Lets plot it:

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS)
XX = np.arange(0.0, 10.0, 0.1)
yy = clf.intercept_[0] + clf.coef_[0][1]*XX + clf.coef_[0][2]*np.power(XX, 2)
plt.plot(XX, yy, '-r')
plt.title("Engine Size vs CO2 Emission")
plt.xlabel("Engine Size")
plt.ylabel("CO2 Emission")
plt.savefig("../results/regression/Engine_Size_vs_CO2_Emission_poly.png")
plt.show()


# Todo: Evaluation
from sklearn.metrics import r2_score

test_x_poly = poly.fit_transform(test_x)
test_y_predicted = clf.predict(test_x_poly)

# Calculate the metrics
mean_abs_error = np.mean(np.absolute(test_y_predicted - test_y))
mean_sqr_error = np.mean((test_y_predicted - test_y) ** 2)
r2 = r2_score(test_y, test_y_predicted)

print("Mean Absolute Error: {:.2f}".format(mean_abs_error))
print("Residual sum of squares (MSE): {:.2f}".format(mean_sqr_error))
print("R2-score: {:.2f}".format(r2))

# Output:
# Mean Absolute Error: 21.74
# Residual sum of squares (MSE): 769.10
# R2-score: 0.79
