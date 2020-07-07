# Project Title

## Predict density of wine based on its acidity using Locally weighted Linear regression

In this problem, we will generalize the ideas of linear regression to implement locally weighted linear regression where we want to "weigh" different training examples differently.

* We want to minimize the following error function: J_theta = 1/2(sum_over_all_m(w(i)\*(y(i) - Theta.T\*X(i))))
* In the matrix notation, the error function above can alternately be written as: J_theta = 1/2\*(X\*Theta - Y).T\*W\*(X\*Theta - Y)
* The value of theta that minimizes J_thata is given by inverse((X.T\*X))\*X.T\*Y . By finding the derivative of J_theta and setting that to zero, we can generalize the normal equation to the weighted setting above.

The files "linearX.csv"and "linearY.csv"contain the acidity of the wine (x(i)'s) and its density (y(i)'s) respectively, with one training example per row.

* Will be implementing Unweighted least squares linear regression first using normal equation to learn the relation-ship between x(i)'s and y(i)'s.
* Will be implementing locally weighted linear regression on this dataset using the weighted normal equations written above to learn the relation-ship between x(i)'s and y(i)'s.
* When evaluating h_theta at a query point x, we use weights: w(i) = exp(- ((x - (x(i)))^2)/2*tau^2) where tau is the bandwidth parameter
* Will plot the data on a two-dimensional graph and plot the hypothesis function learned by the Weighted and Unweighted Linear Regression algorithm.
