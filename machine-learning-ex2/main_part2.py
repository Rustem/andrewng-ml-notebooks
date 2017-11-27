import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def sigmoid(z):
    g = np.zeros(z.shape)
    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the sigmoid of each value of z (z can be a matrix,
    #               vector or scalar).





    # =============================================================
    return g


def costReg(theta, X, y, learningRate):
    # convert to numpy matrix
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    # You need to return the following variables correctly
    J = 0

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.






    # =============================================================
    return J


def gradientReg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute gradient





    # =============================================================

    return grad


def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]


def main():
    print("***********************START**********************\n")
    print("***********************PART 2**********************\n")
    # reading a dataset file
    path = 'ex2data2.txt'
    data2 = pd.read_csv(path, header=None, names=['Test 1', 'Test 2', 'Accepted'])
    #-------------------------------------------------------------

    # print the first raws of the data
    print("The first several raws of our dataset:")
    print(data2.head())
    input("\n1. Press enter to continue...")
    #-------------------------------------------------------------

    # Plot the entire dataset
    positive = data2[data2['Accepted'].isin([1])]
    negative = data2[data2['Accepted'].isin([0])]
    fig, ax = plt.subplots(figsize=(12,8))
    ax.scatter(positive['Test 1'], positive['Test 2'], s=50, c='b', marker='o', label='Accepted')
    ax.scatter(negative['Test 1'], negative['Test 2'], s=50, c='r', marker='x', label='Rejected')
    ax.legend()
    ax.set_xlabel('Test 1 Score')
    ax.set_ylabel('Test 2 Score')
    plt.show()
    #-------------------------------------------------------------

    # Adding polinomial features
    degree = 5
    x1 = data2['Test 1']
    x2 = data2['Test 2']
    data2.insert(3, 'Ones', 1)
    for i in range(1, degree):
        for j in range(0, i):
            data2['F' + str(i) + str(j)] = np.power(x1, i-j) * np.power(x2, j)

    data2.drop('Test 1', axis=1, inplace=True)
    data2.drop('Test 2', axis=1, inplace=True)
    print('Data sample:')
    print(data2.head())
    input("\n2. Press enter to continue...")
    #-------------------------------------------------------------

    # set X and y (remember from above that we moved the label to column 0)
    cols = data2.shape[1]
    X2 = data2.iloc[:,1:cols]
    y2 = data2.iloc[:,0:1]
    #-------------------------------------------------------------

    # convert to numpy arrays and initalize the parameter array theta
    X2 = np.array(X2.values)
    y2 = np.array(y2.values)
    theta2 = np.zeros(11)
    #-------------------------------------------------------------

    # Check regularized cost function with initial theta
    learningRate = 1
    print('\n\n Cost at zero initialization: {0:.2f}'.format(costReg(theta2, X2, y2, learningRate)))
    input("\n3. Press enter to continue...")
    #-------------------------------------------------------------

    # Gradient Descent Optimization
    import scipy.optimize as opt
    result2 = opt.fmin_tnc(func=costReg, x0=theta2, fprime=gradientReg, args=(X2, y2, learningRate))
    print('Final theta values:')
    print(result2)
    input("\n4. Press enter to continue...")
    #-------------------------------------------------------------

    # output results
    theta_min = np.matrix(result2[0])
    predictions = predict(theta_min, X2)
    correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y2)]
    accuracy = (sum(map(int, correct)) % len(correct))
    print('\naccuracy = {0}%'.format(accuracy))
    input("\n5. Press enter to continue...")
    #-------------------------------------------------------------

    print("\n***********************FINISH**********************\n")


if __name__ == '__main__':
    main()