from sympy import sympify, diff, lambdify, solve, simplify
from sympy.abc import t
import math
from math import factorial, comb
import numpy as np

func = sympify(input())
F = lambdify(t, func)
num_samples = int(input())
start, end = [float(x) for x in input().split()]
df = diff(func, t)
sln = solve(simplify(df), t)

max_value = -math.inf
for v in sln:
    v = float(v)
    if start <= v <= end:
        max_value = max(max_value, F(float(v)))

max_value = max(max_value, F(start))
max_value = max(max_value, F(end))

x_samples = np.random.uniform(start, end, (num_samples, ))
y_samples = np.random.uniform(0, max_value, (num_samples, ))

num_correct_samples = 0
for i in range(num_samples):
    if y_samples[i] <= F(x_samples[i]):
        num_correct_samples += 1


print(np.round((num_correct_samples / num_samples) * max_value * (end - start), 2))