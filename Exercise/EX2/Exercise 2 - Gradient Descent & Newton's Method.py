import numpy as np
import math

# Implement a gradient descent algorithm to find the minimum of this function.
# Your algorithm should start with a learning rate of 1, and then implement the "bold driver" heuristic.
# That is, if the value of the objective function increases in an iteration, then you half the learning rate.
# If it decreases, multiply the learning rate by 1.1.
#
# The algorithm should be encapsulated within a procedure gd_optimize (a), where a is a NumPy array with two entries (the starting x and starting y).
#
# At the end of each iteration, your code should print out the value of the objective function.
# When the change in the objective function is less than 10e-20, the code should print out the current value for x and y and then exit.
#

# the functions to calculate gradient.


def dx(x, y):
    constCos = math.cos(x + y)
    return constCos + 2 * (x - y) - 1.5


def dy(x, y):
    constCos = math.cos(x + y)
    return constCos - 2 * (x - y) + 2.5


def dxdy(x, y):
    return -math.sin(x + y) - 2


def secondDerivative(x, y):
    return -math.sin(x + y) + 2

# ---------------------------------------------


def f(x, y):
    return math.sin(x + y) + (x - y) ** 2 - 1.5 * x + 2.5 * y + 1


def gd_optimize(arr):
    x, y = arr[0], arr[1]
    cur = f(x, y)
    learning_rate = 1.0

    print(cur)
    e = float('inf')

    while e > 10e-20:

        # constCos = math.cos(x + y)
        Nx = x - learning_rate * (math.cos(x + y) + 2 * (x - y) - 1.5)
        y = y - learning_rate * (math.cos(x + y) - 2 * (x - y) + 2.5)
        x = Nx
        pre = cur
        cur = f(x, y)
        e = abs(cur - pre)
        # print(x, y)
        print(cur)

        if pre < cur:
            learning_rate *= 0.5
        else:
            learning_rate *= 1.1
    # print(x, y)
    return np.array([x, y])


def hessianMatrix(x, y):
    return np.array([[secondDerivative(x, y), dxdy(x, y)], [dxdy(x, y), secondDerivative(x, y)]])


def nm_optimize(arr):
    x, y = arr
    cur = f(x, y)
    print(cur)
    e = float('inf')

    while e > 10e-20:
        darr = np.array([dx(x, y), dy(x, y)])
        gradient = np.dot(np.linalg.inv(hessianMatrix(x, y)), darr)
        # print(x, y, np.linalg.inv(hessianMatrix(x, y)), darr, gradient)
        x -= gradient[0]
        y -= gradient[1]
        pre = cur
        cur = f(x, y)
        print(cur)
        e = abs(cur - pre)

    return np.array([x, y])


ㄓ

print(gd_optimize(np.array([-0.2, - 1.0])))
print(nm_optimize(np.array([-0.2, -1.0])))
