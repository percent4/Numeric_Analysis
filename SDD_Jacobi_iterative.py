import numpy as np

# check if matrix A is an SDD
# return: boolean value
def isSDD(A):
    for i in range(A.shape[0]):
        if(2*abs(A[i,i]) <= sum(abs(A[i,:]))):
            return False

    return True

# A = D+L+U
# D: diagonal part of A
# L: upper triangular of A
# U: lower triangular of A
# return: if A is SDD, returns a list of three elements which is the decomposition of A
#         else, returns a list of only one element: False
def decomposition(A):
    if isSDD(A):
        n = A.shape[0]    # shape of matrix A
        D = np.array([[0] * n] * n, dtype='float64')
        U = np.array([[0] * n] * n, dtype='float64')
        for i in range(n):
            D[i,i] = A[i,i]
            for j in range(i):
                U[i,j] = A[i,j]
        L = A-D-U

        return [D, L, U]

    else:
        print('Matrix A is not an SDD.')
        return [False]

# using Jacobi iteration to solve linear equation
# the linear equation: Ax=b
# default iteration times is 100, which can be adjusted
# return: solution of linear equation or None
def solve_linear_equation(A, b, iter_time=100):
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
        # x_k+1 = D_inv*((L+U)x_k+b)
        for k in range(iter_time):    # iteration times
            x = np.dot(D_inv, np.dot(-(L+U), x)+b)

        return x    # solution

def main():
    A = np.array([[4, 1, 0, 1],
                  [1, 3, 1, 0],
                  [2, 3, 7, 1],
                  [2, 5, 1, 9]
                 ])
    b = np.array([5, 6, 19, 9])
    sol = solve_linear_equation(A, b)
    if not sol is None:
        print('The solution of equation Ax=b is %s'%sol)

main()