# Implementation of Non-Linear Regression using Scikit-learn
# Create a model, train, test, and use the model
# Data: china_gdp.csv

# Todo: Importing required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Todo: Understanding Data
# china_gdp.csv
# For an example, we're going to try and fit a non-linear model to the datapoints corresponding to China's GDP from 1960 to 2014.
# We download a dataset with two columns, the first, a year between 1960 and 2014,
# the second, China's corresponding annual gross domestic income in US dollars for that year.


# Todo: Reading the data in
df = pd.read_csv("../data/china_gdp.csv")
# print(df.head(10))
# print(df.shape)

# Output:
#    Year         Value
# 0  1960  5.918412e+10
# 1  1961  4.955705e+10
# 2  1962  4.668518e+10
# 3  1963  5.009730e+10
# 4  1964  5.906225e+10


# Todo: Plot the dataset
# plt.figure(figsize=(8,5))
x_data, y_data = (df["Year"].values, df["Value"].values)
# plt.plot(x_data, y_data, 'o')
# plt.title("China's Annual Gross Domestic Income (1960-2014)")
# plt.ylabel('GDP (US dollar)')
# plt.xlabel('Year')
# plt.savefig("../results/regression/china_gdp_1960-2014.png")
# plt.show()


# This is what the datapoints look like.
# It kind of looks like an either logistic or exponential function.
# The growth starts off slow, then from 2005 on forward, the growth is very significant.
# And finally, it decelerate slightly in the 2010s.


# Todo: Select a Model
# From an initial look at the plot, we determine that the logistic function could be a good approximation,
# since it has the property of starting with a slow growth, increasing growth in the middle,
# and then decreasing again at the end; as illustrated below:

# X = np.arange(-5.0, 5.0, 0.1)
# Y = 1.0 / (1.0 + np.exp(-X))
#
# plt.plot(X,Y)
# plt.ylabel('Dependent Variable')
# plt.xlabel('Independent Variable')
# plt.show()

# The formula for the logistic function is the following:
# 𝑌̂ =1/(1+𝑒𝛽1(𝑋−𝛽2))
# 𝛽1 : Controls the curve's steepness,
# 𝛽2 : Slides the curve on the x-axis.

# Todo: Building The Model
# Build regression model and initialize its parameters
def sigmoid(x, beta_1, beta_2):
    y = 1 / (1 + np.exp(-beta_1*(x-beta_2)))
    return y


# Todo: Lets look at a sample sigmoid line that might fit with the data
beta_1 = 0.10
beta_2 = 1990.0

# logistic function
y_predicted = sigmoid(x_data, beta_1, beta_2)

# plot initial prediction against datapoints
# plt.plot(x_data, y_predicted*15000000000000.)
# plt.plot(x_data, y_data, 'ro')
# plt.savefig("../results/regression/china_gdp_sigmoide.png")
# plt.show()

# Todo: Our task here is to find the best parameters for our model. Lets first normalize our x and y
# Lets normalize our data
xdata =x_data/max(x_data)
ydata =y_data/max(y_data)

# How we find the best parameters for our fit line?
# we can use curve_fit which uses non-linear least squares to fit our sigmoid function, to data.
# Optimal values for the parameters so that the sum of the squared residuals of sigmoid(xdata, *popt) - ydata is minimized.
# popt are our optimized parameters.

from scipy.optimize import curve_fit
popt, pcov = curve_fit(sigmoid, xdata, ydata)

# print the final parameters
print("beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))

# Output: beta_1 = 690.453018, beta_2 = 0.997207

# Todo: Plot our resulting regression model
x = np.linspace(1960, 2015, 55)
x = x/max(x)
plt.figure(figsize=(8, 5))
y = sigmoid(x, *popt)
plt.plot(xdata, ydata, 'ro', label='data')
plt.plot(x, y, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.savefig("../results/regression/china_gdp_data_model.png")
plt.show()

# Todo: Calculate the accuracy of the model
# Todo: split data into train/test
msk = np.random.rand(len(df)) < 0.8
train_x = xdata[msk]
test_x = xdata[~msk]
train_y = ydata[msk]
test_y = ydata[~msk]

# build the model using train set
popt, pcov = curve_fit(sigmoid, train_x, train_y)

# predict using test set
y_hat = sigmoid(test_x, *popt)

# Todo: evaluation
print("Mean absolute error: %.2f" % np.mean(np.absolute(y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((y_hat - test_y) ** 2))

from sklearn.metrics import r2_score
print("R2-score: %.2f" % r2_score(y_hat , test_y) )

# Output:
# Mean absolute error: 0.04
# Residual sum of squares (MSE): 0.00
# R2-score: 0.96

