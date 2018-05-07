# using Jacobi and Gauss-Seidel and SOR Iterative Method to solve linear equation Ax=b.
import numpy as np
from numpy.linalg import inv

# check if matrix A is an SDD
# return: boolean value
def isSDD(A):
    for i in range(A.shape[0]):
        if(2*abs(A[i,i]) <= sum(abs(A[i,:]))):
            return False

    return True

# A = D+L+U
# D: diagonal part of A
# U: upper triangular of A
# L: lower triangular of A
# return: if A is SDD, returns a list of three elements which is the decomposition of A
#         else, returns a list of only one element: False
def decomposition(A):
    if isSDD(A):
        n = A.shape[0]    # shape of matrix A
        D = np.array([[0] * n] * n, dtype='float64')
        L = np.array([[0] * n] * n, dtype='float64')
        for i in range(n):
            D[i,i] = A[i,i]
            for j in range(i):
                L[i,j] = A[i,j]
        U = A-D-L

        return [D, L, U]

    else:
        print('Matrix A is not an SDD.')
        return [False]

# using Jacobi iteration to solve linear equation
# the linear equation: Ax=b
# default iteration times is 100, which can be adjusted
# return: solution of linear equation or None
def solve_linear_equation_Jacobi(A, b, iter_time=100):
    lst = decomposition(A)
    if len(lst) == 3:   # if A is an SDD
        [D, L, U] = lst   # decomposition of matrix A
        n = A.shape[0]
        x = np.array([0]*n, dtype='float64')    # initial solution of Ax=b

        # inverse of matrix D
        D_inv = np.array([[0] * n] * n, dtype='float64')
        for i in range(n):
            D_inv[i,i] = 1/(D[i,i])

        # Jacobi iteration
        # x_k+1 = D_inv*(-(L+U)x_k+b)
        for k in range(iter_time):    # iteration times
            x = np.dot(D_inv, np.dot(-(L+U), x) + b)

        return x    # solution

# using Gauss-Seidel iteration to solve linear equation
# the linear equation: Ax=b
# default iteration times is 100, which can be adjusted
# return: solution of linear equation or None
def solve_linear_equation_GS(A, b, iter_time=100):
    lst = decomposition(A)
    if len(lst) == 3:  # if A is an SDD
        [D, L, U] = lst  # decomposition of matrix A
        n = A.shape[0]
        x = np.array([0] * n, dtype='float64')  # initial solution of Ax=b

        # Gauss-Seidel iteration
        # x_k+1 = (L+D)_inv*(-Ux_k+b)
        for k in range(iter_time):  # iteration times
            x = np.dot(inv(L+D), np.dot(-U, x) + b)

        return x  # solution

# using SOR iteration to solve linear equation
# the linear equation: Ax=b
# parameters: A, b, w, iter_time
# w: relaxation parameter, 0 <= w <= 2.
# default iteration times is 100, which can be adjusted
# return: solution of linear equation or None
def solve_linear_equation_SOR(A, b, w, iter_time=100):
    lst = decomposition(A)
    if len(lst) == 3:  # if A is an SDD
        [D, L, U] = lst  # decomposition of matrix A
        n = A.shape[0]
        x = np.array([0] * n, dtype='float64')  # initial solution of Ax=b

        # SOR iteration
        # x_k+1 = (wL+D)_inv*((1-w)Dx_k-wUx_k+wb)
        for k in range(iter_time):  # iteration times
            x = np.dot(inv(w*L+D), np.dot((1-w)*D, x)+np.dot(-w*U, x) + w*b)

        return x  # solution

# main function
def main():
    A = np.array([[3, -1, 0, 0, 0, 0.5],
                  [-1, 3, -1, 0, 0.5, 0],
                  [0, -1, 3, -1, 0, 0],
                  [0, 0, -1, 3, -1, 0],
                  [0, 0.5, 0, -1, 3, -1],
                  [0.5, 0, 0, 0, -1, 3]
                 ])
    b = np.array([2.5, 1.5, 1, 1, 1.5, 2.5])
    sol1 = solve_linear_equation_Jacobi(A, b, 6)
    sol2 = solve_linear_equation_GS(A, b, 6)
    sol3 = solve_linear_equation_SOR(A, b, 1.1, 6)
    if not sol1 is None and not sol2 is None:
        print('The solution of equation Ax=b by Jacobi is %s'%sol1)
        print('The solution of equation Ax=b by GS is %s' % sol2)
        print('The solution of equation Ax=b by SOR is %s' % sol3)

main()
