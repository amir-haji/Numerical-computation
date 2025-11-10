from sympy import sympify, diff, solve, simplify, integrate
from sympy.utilities import lambdify
from sympy.abc import x
import math
from math import factorial, comb
import numpy as np

func = sympify(input())
F = lambdify(x, func, 'numpy')
n = int(input())
start, end = [float(x) for x in input().split()]

real_value = float(integrate(func, (x, start, end)))

h = (end - start) / n
x = np.array([start + i*h for i in range(n + 1)])
h2 = (end - start) / (2 * n)
x2 = np.array([start + i*h2 for i in range(2 * n + 1)])
y = F(x)
y2 = F(x2)

rectangular_value = 0
for i in range(n):
    rectangular_value += F(x[i]) * h

midpoint_value = 0
for i in range(n):
    midpoint_value += F((x[i] + x[i+1]) / 2) * h

trapezoidal_value = 0
for i in range(n):
    trapezoidal_value += (h / 2) * (y[i] + y[i + 1])


sympson_value = 0
for i in range(1, n + 1):
    sympson_value += (h2 / 3) * (y2[2 * i - 2] + 4 * y2[2 * i - 1] + y2[2 * i])


for value in [rectangular_value, midpoint_value, trapezoidal_value, sympson_value]:
    print(f'{np.round(value, 3)} {np.round(np.abs(value - real_value), 3)}')
