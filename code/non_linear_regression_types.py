# Implementation of Non-Linear Regression using Scikit-learn

# Todo: Importing required packages
import numpy as np
import matplotlib.pyplot as plt


# Explanation: Linear Regression and Non-Linear Regression
# Though Linear regression is very good to solve many problems, it cannot be used for all datasets.
# First recall how linear regression, could model a dataset.
# It models a linear relation between a dependent variable y and independent variable x.
# It had a simple equation, of degree 1, for example y =  2𝑥  + 3.


# x = np.arange(-5.0, 5.0, 0.1)

# You can adjust the slope and intercept to verify the changes in the graph
# y = 2*(x) + 3
# y_noise = 2 * np.random.normal(size=x.size)
# y_data = y + y_noise

# plt.figure(figsize=(8,6))
# plt.plot(x, y_data,  'bo')
# plt.plot(x,y, 'r')
# plt.ylabel('Dependent Variable')
# plt.xlabel('Independent Variable')
# plt.show()


# Non-linear regressions are a relationship between independent variables  𝑥  and a dependent variable  𝑦
# which result in a non-linear function modeled data. Essentially any relationship that is not linear can be
# termed as non-linear, and is usually represented by the polynomial of  𝑘  degrees (maximum power of  𝑥 ).
# 𝑦=𝑎𝑥3+𝑏𝑥2+𝑐𝑥+𝑑
# Non-linear functions can have elements like exponential, logarithm, fraction, and others. For example:
# 𝑦=log(𝑥)
# Or even, more complicated such as :
# 𝑦=log(𝑎𝑥3+𝑏𝑥2+𝑐𝑥+𝑑)

# Todo: cubic function's graph.

# x = np.arange(-5.0, 5.0, 0.1)

# You can adjust the slope and intercept to verify the changes in the graph
# y = 1*(x**3) + 1*(x**2) + 1*x + 3
# y_noise = 20 * np.random.normal(size=x.size)
# y_data = y + y_noise
# plt.plot(x, y_data,  'bo')
# plt.plot(x,y, 'r')
# plt.ylabel('Dependent Variable')
# plt.xlabel('Independent Variable')
# plt.show()

# As you can see, this function has  𝑥3  and  𝑥2  as independent variables.
# Also, the graphic of this function is not a straight line over the 2D plane.
# So this is a non-linear function.
# Some other types of non-linear functions are:

# Todo: Quadratic: 𝑌=𝑋2

# x = np.arange(-5.0, 5.0, 0.1)

# You can adjust the slope and intercept to verify the changes in the graph

# y = np.power(x,2)
# y_noise = 2 * np.random.normal(size=x.size)
# y_data = y + y_noise
# plt.plot(x, y_data,  'bo')
# plt.plot(x,y, 'r')
# plt.ylabel('Dependent Variable')
# plt.xlabel('Independent Variable')
# plt.show()

# todo: Exponential
# An exponential function with base c is defined by: 𝑌=𝑎+𝑏𝑐𝑋
# where b ≠0, c > 0 , c ≠1, and x is any real number.
# The base, c, is constant and the exponent, x, is a variable.

# X = np.arange(-5.0, 5.0, 0.1)

# You can adjust the slope and intercept to verify the changes in the graph
# Y = np.exp(X)
#
# plt.plot(X,Y)
# plt.ylabel('Dependent Variable')
# plt.xlabel('Independent Variable')
# plt.show()

# Todo: Logarithmic
# The response  𝑦  is a results of applying logarithmic map from input  𝑥 's to output variable  𝑦 .
# It is one of the simplest form of log(): i.e. 𝑦=log(𝑥)
# Please consider that instead of  𝑥 , we can use  𝑋 , which can be polynomial representation of the  𝑥 's.
# In general form it would be written as: 𝑦=log(𝑋)

# X = np.arange(-5.0, 5.0, 0.1)
#
# Y = np.log(X)
#
# plt.plot(X,Y)
# plt.ylabel('Dependent Variable')
# plt.xlabel('Independent Variable')
# plt.show()

# Todo: Sigmoidal/Logistic: 𝑌 = 𝑎 + 𝑏/(1+𝑐(𝑋−𝑑))

X = np.arange(-5.0, 5.0, 0.1)


Y = 1-4/(1+np.power(3, X-2))

plt.plot(X, Y)
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()
