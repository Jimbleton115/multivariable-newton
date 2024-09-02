import numpy as np
import math

def f_1(X):
    return 2*X[0]*X[0] + 3*X[1]

def f_2(X):
    return 3*X[0] + 3*X[1] + 3

def f_1_partial_x0(X):
    return 4*X[0]

def f_1_partial_x1(X):
    return 3

def f_2_partial_x0(X):
    return 3

def f_2_partial_x1(X):
    return 3

def multivariable_newton(X,F,n,J,N):
    """
    Implements the multivariable newton method.
    Input:
        Initial guess X (numpy array)
        functions f_1,...,f_n in a numpy array F
        Number of variables/functions n
        Jacobian J
        max iterations N
    Output:
        the x value for an approximate 0
    """
    # run for N iterations
    for i in range(1,N+1):
        # initialize M matrix that will hold the evaluated Jacobian
        M = np.zeros((n,n))
        # Evaluate the Jacobian at X
        for j in range(0,n):
            for k in range(0,n):
                M[j][k] = J[j][k](X)
        # Invert the Jacobian
        M_inverse = np.linalg.inv(M)
        # Initialize F_eval matrix that will hold F evaluated at X
        F_eval = np.zeros(n)
        # Evaluate F at X
        for j in range(0,n):
            F_eval[j] = F[j](X)
        # Multiply the Jacobian and F
        H = np.matmul(M_inverse,F_eval)
        # Iterate X following the method
        X = np.subtract(X,H)
    return X

def main():
    F = np.array([f_1,f_2])
    X = np.array([-1,-1])
    n = 2
    J = np.array([[f_1_partial_x0,f_1_partial_x1],[f_2_partial_x0,f_2_partial_x1]])

    N = 10

    X = multivariable_newton(X,F,n,J,N)

    print(X)

main()