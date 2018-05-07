# solve linear equation using Cholesky decomposition

from maxtrix_decomposition import Cholesky
import numpy as np
from copy import deepcopy
# solve linear equation AX=B
def sol(a, b):
    r = Cholesky(a)

    z = []
    for k in range(len(b)):
        t = (b[k]-sum([z[i]*r.T[k, i] for i in range(k)]))/r.T[k, k]
        z.append(t)

    x = [0]*len(b)
    for k in range(len(b)-1, -1, -1):
        t = (z[k]-sum([r[k, i]*x[i] for i in range(len(x)-1, k, -1)]))/r[k, k]
        x[k] = t

    return x

def main():

    a = np.array([
        [4, 0.5, 1],
        [0.5, 1.0625, 0.25],
        [1, 0.25, 0.515625],
    ], dtype='float64')
    A = deepcopy(a)
    b = [10, 20, 30]
    x = np.mat(sol(a, b))

    print('The solution is:\n', x)
    # check whether ax == b

    print(A.dot(x.T).T)
    
main()
