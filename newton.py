# using Newton's Method to solve nonlinear equation
# Newton's Method use the function's derivative
# Newton's Method depends on initial value(s)

import math
import numpy as np
from numpy.linalg import inv

'''
    f: original function
    df: original function's derivative
    x: initial value
    return: x, final solution
'''
def single_variable_Newton(f, df, x, iter_time=10):
    # Newton iteration: x_k+1 = x_k - f(x_k)/df(x_k)
    for _ in range(iter_time):    # iteration times
        x = x-f(x)/df(x)

    return x

# try to concise the parameters of the function
# this def is valid for 2*2 equations, 2 funcitons, with each two variables
def multi_variables_Newton(f,g,df_u,df_v,dg_u,dg_v, X, iter_time):
    # Newton iteration: DF(X_k)s=-F(X_k)
    #                   X_k+1 = X_k+s

    for _ in range(iter_time):    # iteration times
        F = np.array([f(*X), g(*X)])
        DF = np.array([[df_u(*X), df_v(*X)],
                       [dg_u(*X), dg_v(*X)]
                      ])
        # here use the inverse of DF, which is not wise in lagre-scale problem
        # use Gauss Elimination Method to replace inverse if possible
        # if DF can be decomposed of LU, use it to solve the linear equation
        # if DF is SPD, try Cholesky decomposition to solve the linear equation
        # if DF is SDD, try Jacobi or Gauss-Seidei Iteration Method
        s = np.dot(inv(DF), -F)
        X += s

    return X

def main():
    '''
    f = lambda x: x ** 3 + x - 1
    df = lambda x: 3 * x ** 2 + 1
    x = -0.7
    '''
    f = lambda x: math.sin(x) - math.cos(x)
    df = lambda x: math.cos(x) + math.sin(x)
    x = 0
    sol = single_value_Newton(f, df, x, 100)
    print(sol)


    '''
        equations: v-u**3 = 0
                  u**2+v**2-1 = 0
    '''
    # origiral functions
    f = lambda u,v: v-u**3
    g = lambda u,v: u**2+v**2-1

    # derivative function to form Jacobi Matrix
    df_u = lambda u,v: -3**u*2
    df_v = lambda u,v: 1
    dg_u = lambda u,v: 2*u
    dg_v = lambda u,v: 2*v

    # Initial solution vector
    X = np.array([1,2], dtype='float64')

    SOL = multi_values_Newton(f, g, df_u, df_v, dg_u, dg_v, X, 10)
    print(SOL)

    '''
            equations: 6*u**3+u*v-3*v**3-4 = 0
                      u**2-18*u*v**2+16*v**3+1 = 0
                      
            this equations have 3 solutions
    '''
    # origiral functions
    f = lambda u, v: 6*u**3+u*v-3*v**3-4
    g = lambda u, v: u**2-18*u*v**2+16*v**3+1

    # derivative function to form Jacobi Matrix
    df_u = lambda u, v: 18*u**2+v
    df_v = lambda u, v: u-9*v**2
    dg_u = lambda u, v: 2*u-18*v**2
    dg_v = lambda u, v: -36*u*v+48*v**2

    # Initial solution vector
    # X = np.array([2, 2], dtype='float64')
    # X = np.array([0.1, 0.1], dtype='float64')
    X = np.array([0.8, 0.4], dtype='float64')

    SOL = multi_values_Newton(f, g, df_u, df_v, dg_u, dg_v, X, 50)
    print(SOL)

main()