# -*- coding: utf-8 -*-

'''
Gradient Descent Examples

function: f = x**4+y**2+1
initial point: (2,3)
step size: 0.1
'''

import math

# Eculid distance
def norm_2(*args):
    sum = 0
    for i in args:
        sum += i**2
    return math.sqrt(sum)

def gd(func, grad_func, initial, step = 0.1, eg = 0.001, ex = 0.001, N = 1000):
    count = 1

    for _ in range(N):

        derivate = []
        for i in range(len(initial)):
            initial[i] -= step * grad_func[i](initial[i])
            derivate.append(grad_func[i](initial[i]))


        if norm_2(*initial) <= ex or norm_2(*derivate) <= eg:
            print('Iteration counts:%d' % count)
            return initial

        count += 1

    print('Iteration counts:%d' % count)
    return initial

def main():

    func = lambda x,y: x**4 + y**2 + 1

    grad_func = [lambda x: 4*x**3, lambda y: 2*y]

    initial = [2,3]

    res = gd(func, grad_func,  initial)

    print(res)

main()

