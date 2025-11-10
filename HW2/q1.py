from sympy import sympify, diff, solve, simplify, integrate
from sympy.utilities import lambdify
from sympy.abc import x
import math
from math import factorial, comb
import numpy as np

n = int(input())
X = np.array([float(x) for x in input().split()])
y = np.array([float(x) for x in input().split()])

interpolation = 0
for i in range(n):
    coeff = 1
    for j in range(n):
        if i != j:
            coeff *= (x - X[j]) / (X[i] - X[j])

    interpolation += coeff * y[i]


d = 0
int = diff(interpolation, x)
for i in range(n):
    func = lambdify(x, int)

    c = 0
    for a in X:
        if abs(func(a)) < (10 ** -6):
            c += 1

    if c == len(X):
        d = i
        break
    int = diff(int, x)

print(d)
