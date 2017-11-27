import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sigmoid(z):
    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the sigmoid of each value of z (z can be a matrix,
    #               vector or scalar).
    """
    return: computed sigmoid of z (in our case array)
    """
    return 1 / (1 + np.exp(-z))


def cost(theta, X, y):
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
    


    l = len(X)
    # NOTE: see equation in the task for cost function of sigmoid
    first_term = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second_term = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    J = np.sum(first_term - second_term) / l
    return J


def gradient(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    # You need to return the following variables correctly
    grad = np.zeros(parameters)

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute gradient


    # =============================================================
    error_value = sigmoid(X * theta.T) - y
    
    for i in range(parameters):
        term = np.multiply(error_value, X[:,i])   # multiply value of ith feature by error (see equation of sigmoid gradient)
        grad[i] = np.sum(term) / len(X)   # calculate gradient vector of coordinate at base (x_i)
    return grad


def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]


def main():
    print("***********************START**********************\n")
    print("***********************PART 1**********************\n")
    # reading a dataset file
    path = 'ex2data1.txt'
    data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
    #-------------------------------------------------------------

    # print the first raws of the data
    print("The first several raws of our dataset:")
    print(data.head())
    input("\n1. Press enter to continue...")
    #-------------------------------------------------------------

    # Plot the entire dataset
    positive = data[data['Admitted'].isin([1])]
    negative = data[data['Admitted'].isin([0])]
    fig, ax = plt.subplots(figsize=(12,8))
    ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
    ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
    ax.legend()
    ax.set_xlabel('Exam 1 Score')
    ax.set_ylabel('Exam 2 Score')
    plt.show()
    #-------------------------------------------------------------

    # Check the sigmoid function
    nums = np.arange(-10, 10, step=1)
    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(nums, sigmoid(nums), 'r')
    plt.show()
    #-------------------------------------------------------------

    # add a ones column - this makes the matrix multiplication work out easier
    data.insert(0, 'Ones', 1)
    # set X (training data) and y (target variable)
    cols = data.shape[1]
    X = data.iloc[:,0:cols-1]
    y = data.iloc[:,cols-1:cols]
    # convert to numpy arrays and initalize the parameter array theta
    X = np.array(X.values)
    y = np.array(y.values)
    theta = np.zeros(3)
    print('\n\n Cost at zero initialization: {0:.2f}'.format(cost(theta, X, y)))
    input("\n2. Press enter to continue...")
    #-------------------------------------------------------------

    # Gradient Descent Optimization
    import scipy.optimize as opt
    result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))
    print('Final theta values:')
    print(result)
    input("\n3. Press enter to continue...")
    #-------------------------------------------------------------

    # output results
    theta_min = np.matrix(result[0])
    predictions = predict(theta_min, X)
    correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
    accuracy = (sum(map(int, correct)) % len(correct))
    print('\naccuracy = {0}%'.format(accuracy))
    input("\n4. Press enter to continue...")
    #-------------------------------------------------------------

    print("\n***********************FINISH**********************\n")


if __name__ == '__main__':
    main()