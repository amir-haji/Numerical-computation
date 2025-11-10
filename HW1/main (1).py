from sympy import sympify, diff, lambdify
from sympy.abc import x, y 
from math import factorial, comb
import numpy as np

func = sympify(input())
F = lambdify([x, y], func)
x0, y0 = [float(x) for x in input().split(' ')]
x1, y1 = [float(x) for x in input().split(' ')]
n = int(input())

total_sum = 0
for d in range(0, n+1):

    sum = 0
    for i in range(0, d+1):
        g = func.copy()
        g_1 = diff(g, x, i)
        g_2 = diff(g_1, y, d-i)

        G_2 = lambdify([x, y], g_2)

        sum += comb(d, i) * G_2(x0, y0) * ((x1 - x0)**i) * ((y1 - y0) ** (d-i))

    total_sum += (1/factorial(d)) * sum

print(np.round(total_sum, 4))
print(np.round(F(x1, y1), 4))