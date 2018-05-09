# use Thomas Method to solve tridiagonal linear equation
# algorithm reference: https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm

import numpy as np

# parameter: a,b,c,d are list-like of same length
# tridiagonal linear equation: Ax=d
# b: main diagonal of matrix A
# a: main diagonal below of matrix A
# c: main diagonal upper of matrix A
# d: Ax=d
# return: x(type=list), the solution of Ax=d
def TDMA(a,b,c,d):

    try:
        n = len(d)    # order of tridiagonal square matrix

        # use a,b,c to create matrix A, which is not necessary in the algorithm
        A = np.array([[0]*n]*n, dtype='float64')

        for i in range(n):
            A[i,i] = b[i]
            if i > 0:
                A[i, i-1] = a[i]
            if i < n-1:
                A[i, i+1] = c[i]

        # new list of modified coefficients
        c_1 = [0]*n
        d_1 = [0]*n

        for i in range(n):
            if not i:
                c_1[i] = c[i]/b[i]
                d_1[i] = d[i] / b[i]
            else:
                c_1[i] = c[i]/(b[i]-c_1[i-1]*a[i])
                d_1[i] = (d[i]-d_1[i-1]*a[i])/(b[i]-c_1[i-1] * a[i])

        # x: solution of Ax=d
        x = [0]*n

        for i in range(n-1, -1, -1):
            if i == n-1:
                x[i] = d_1[i]
            else:
                x[i] = d_1[i]-c_1[i]*x[i+1]

        x = [round(_, 4) for _ in x]

        return x

    except Exception as e:
        return e

def main():

    a = [0, 1, 1, 1, 1]
    b = [4, 4, 4, 4, 4]
    c = [1, 1, 1, 1, 0]
    d = [1, 0.5, -1, 3, 2]

    '''
    a = [0, 2, 1, 3]
    b = [1, 1, 2, 1]
    c = [2, 3, 0.5, 0]
    d = [2, -1, 1, 3]
    '''

    x = TDMA(a, b, c, d)
    print('The solution is %s'%x)

main()
