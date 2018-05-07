# Cholesky decomposition of matrix

import numpy as np
import math

# the parameter 'a' in functions below is in matrix format
# check if a is a square matrix
def isSquareMatrix(a):
    lengthOfRows = [len(a[i]) for i in range(len(a))]
    if len(set(lengthOfRows)) != 1:
        return False
    else:
        m, n = a.shape
        return m == n

# check if a is a symmetric square matrix
def isSymmetric(a):
    if not isSquareMatrix(a):
        print('The matrix is not a square matrix!')
        return False
    else:
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                if a[i, j] != a.T[i, j]:
                    return False
    return True

# check if a is positive definite
def isPositiveDefinite(a):
    if not isSymmetric(a):
        print('The matrix is not symmetric!')
        return False
    else:
        u, v = np.linalg.eigh(a)
        for val in u:
            if val <= 0:
                return False
    return True

'''
# the Cholesky decomposition of a
def Cholesky(a):
    if not isPositiveDefinite(a):
        print('The matrix is not positive definite!')
        return False
    else:
        a = np.mat(a, dtype='float64')
        n = a.shape[0]
        r = np.mat([[0]*n]*n, dtype='float64')
        for k in range(n):
            r[k, k] = math.sqrt(a[k, k])
            u = a[k, k+1:]/r[k, k]
            r[k, k+1:] = u
            a[k+1:, k+1:] = a[k+1:, k+1:] - u.T.dot(u)

        return r
'''

def Cholesky(a):
    a = np.mat(a, dtype='float64')
    n = a.shape[0]
    r = np.mat([[0] * n] * n, dtype='float64')
    for k in range(n):
        r[k, k] = math.sqrt(a[k, k])
        u = a[k, k + 1:] / r[k, k]
        r[k, k + 1:] = u
        a[k + 1:, k + 1:] = a[k + 1:, k + 1:] - u.T.dot(u)

    return r


# print the Cholesky matrix nicely
def nicePrint(a):
    try:
        a = Cholesky(a)
        n = a.shape[0]
        print('The Cholesky decomposition is:\n')
        for i in range(n):
            for j in range(n):
                print(a[i, j], end='\t')
            print()
    except Exception:
        pass


# main function
def main():
    a = np.array([
                   [1, 1, 1, 1, 1, 1],
                   [1, 2, 3, 4, 5, 6],
                   [1, 3, 6, 10, 15, 21],
                   [1, 4, 10, 20, 35, 56],
                   [1, 5, 15, 35, 70, 126],
                   [1, 6, 21, 56, 126, 252],
              ])

    nicePrint(a)

main()
