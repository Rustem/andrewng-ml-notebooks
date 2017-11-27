import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def computeCost(X, y, theta):
    """
    param X: matrix [lxd] that describes object set with d features
    type X: numpy matrix
    param y: answer vector from learning set
    type y: numpy matrix [lx1]
    param theta: current vector of weights
    type theta: numpy matrix [1x1]

    return: resulting cost that describes the quality of our model
    rtype: float
    """
    # least squares loss function
    l = len(X)  # size of learning set
    inner = np.power(((X * theta.T) - y), 2)  # error vector
    return np.sum(inner) / (2 * l)  # averaging error using values in vector


def gradientDescent(X, y, theta, alpha, iters):
    """
    param X: matrix [lxd] that describes object set with d features
    type X: numpy matrix
    param y: answer vector from learning set
    type y: numpy matrix [lx1]
    param theta: current vector of weights
    type theta: numpy matrix [1x1]
    param alpha: gradient step
    type alpha: numeric
    param iters: max number of iterations
    type iters: int
    return: resulting theta (vector of weights) and cost where algorithm has stopped
    rtype: tuple(numpy matrix, cost array of number of iterations)
   """
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)
    for t in range(iters):
        error = (X * theta.T) - y  # Xw - y

        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))  #
            theta = temp  # update theta vector at item-j
        # compute cost with optimized theta at this step
        cost[t] = computeCost(X, y, theta)
    return theta, cost


def part2():
    print("\n***********************PART 2**********************\n")
    print("**********Linear regression with multiple variables********\n")
    # reading a dataset file
    path = 'ex1data2.txt'
    data2 = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])
    print(data2.head())
    input("\n5. Press enter to continue...")
    #-------------------------------------------------------------

    print("The first several raws of our dataset:")
    data2 = (data2 - data2.mean()) / data2.std()
    print(data2.head())
    input("\n6. Press enter to continue...")
    #-------------------------------------------------------------

    # add ones column
    data2.insert(0, 'Ones', 1)
    # set X (training data) and y (target variable)
    cols = data2.shape[1]
    X2 = data2.iloc[:,0:cols-1]
    y2 = data2.iloc[:,cols-1:cols]
    #-------------------------------------------------------------

    # convert to matrices and initialize theta
    X2 = np.matrix(X2.values)
    y2 = np.matrix(y2.values)
    theta2 = np.matrix(np.array([0,0,0]))
    #-------------------------------------------------------------

    # perform linear regression on the data set
    alpha = 0.01
    iters = 1000
    g2, cost2 = gradientDescent(X2, y2, theta2, alpha, iters)
    #-------------------------------------------------------------

    # get the cost (error) of the model
    print('\n\n Final cost (after gradient descent): {0:.2f}'.format(computeCost(X2, y2, g2)))
    input("\n7. Press enter to continue...")
    #-------------------------------------------------------------

    # Cost function value progress
    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(np.arange(iters), cost2, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Error vs. Training Epoch')
    plt.show()
    #-------------------------------------------------------------
    print("\n***********************FINISH**********************\n")


def main():
    print("***********************START**********************\n")
    print("***********************PART 1**********************\n")
    print("**********Linear regression with one variable********\n")
    # reading a dataset file
    path = 'ex1data1.txt'
    data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
    #-------------------------------------------------------------

    # print the first raws of the data and plot the entire dataset
    print("The first several raws of our dataset:")
    print(data.head())
    input("\n1. Press enter to continue...")
    data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))
    plt.show()
    #-------------------------------------------------------------

    # insert bias unit, i.e. ones
    data.insert(0, 'Ones', 1)
    # split features and labels of the dataset
    cols = data.shape[1]
    X = data.iloc[:,0:cols-1]
    y = data.iloc[:,cols-1:cols]
    # print sample of splitted data
    print("Features and labels:")
    print('{0}'.format(X.head()))
    print('\n\n {0}'.format(y.head()))
    input("\n2. Press enter to continue...")
    #-------------------------------------------------------------

    # MAIN PART
    # convert pandas dataframe to numpy matrix
    X = np.matrix(X.values)
    y = np.matrix(y.values)
    # initial theta value as zeros
    theta = np.matrix(np.array([0,0]))
    print('\n\n Cost at zero initialization: {0:.2f}'.format(computeCost(X, y, theta)))
    input("\n3. Press enter to continue...")
    alpha = 0.01
    iters = 1000
    # gradient descent algorithm
    g, cost = gradientDescent(X, y, theta, alpha, iters)
    # print cost at final stage and plot data
    print('\n\n Final cost: {0:.2f}'.format(computeCost(X, y, g)))
    input("\n4. Press enter to continue...")
    x = np.linspace(data.Population.min(), data.Population.max(), 100)
    f = g[0, 0] + (g[0, 1] * x)
    #--------------------------------------------------------------

    # SHOW FINAL PLOT with prediction line
    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(x, f, 'r', label='Prediction')
    ax.scatter(data.Population, data.Profit, label='Traning Data')
    ax.legend(loc=2)
    ax.set_xlabel('Population')
    ax.set_ylabel('Profit')
    ax.set_title('Predicted Profit vs. Population Size')
    plt.show()
    #---------------------------------------------------------------

    # Cost function value progress
    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(np.arange(iters), cost, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Error vs. Training Epoch')
    plt.show()
    #---------------------------------------------------------------

    # Second part of exercise
    part2()

if __name__ == '__main__':
    main()