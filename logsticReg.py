import math
import pandas as pd
import numpy as np

# the Cholesky decomposition of positive definite matrix
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

# the logistic function
def logistic_func(w, x):
    return 1/(1+math.exp(-np.dot(w, x)))

# read data in csv format
def read_data(file_path):
    data = pd.read_csv(file_path, sep=',')
    return data

# update the weights of w
def update_w(data, w):
    # data_x: features, data_y: labels
    data_x = data.iloc[:, 0:-1]
    m, n = data_x.shape
    data_y = data.iloc[:, -1]

    # the X matrix
    X = np.array([[0]*(n+1)]*m, dtype='float64')
    for i in range(data_x.shape[0]):
        X[i] = [1]+list(data_x.iloc[i, :])

    # the A_T matrix
    A_T = np.array([[0]*m]*m, dtype='float64')
    for i in range(m):
        t = logistic_func(w, X[i])
        A_T[i, i] = t*(1-t)

    # the Hessian matrix
    H_T = np.mat(X).T.dot(np.mat(A_T)).dot(np.mat(X))

    # the U matrix
    temp = np.array([data_y[i]-logistic_func(w, X[i]) for i in range(m)])
    U = np.mat(X.T).dot(np.mat(temp).T)

    # updating weights are solution of H_T*w_delta = U
    w_delta = sol(H_T, U)

    return [w_delta[i][0, 0] for i in range(n+1)]  #return in list format

# calculate the coefficient of logistic regression
def cal_coef(data, cycle_times=10):
    # initialize the weights to be all zeros
    W = [0]*data.shape[1]

    # cycle for cycle_times
    for _ in range(cycle_times):
        # update the weights one by one
        for i in range(len(W)):
            W[i] = W[i]+update_w(data, W)[i]
        print('The weights of %d times:\n%s' % (_+1, W))

    # the final weights: w_final, each number has 4 valid digits
    w_final = [round(i, 6) for i in W]
    print('\nThe final weights after %d iterations is:\n%s'%(cycle_times, w_final))

#main function
def main():
    # file path of csv
    # file_path = 'E://log_reg/USA_vote.csv' # 10 times
    # file_path = 'E://log_reg/my.csv' # 10 times
    file_path = 'E://log_reg/traffic_accident.csv'
    data = read_data(file_path)
    cal_coef(data, 10)

main()