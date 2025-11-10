import numpy as np

degree = int(input())
n = int(input())

x = []
y = []
for _ in range(n):
    a, b = [float(x) for x in input().split(' ')]
    x.append(a)
    y.append(b)

x = np.array(x)
y = np.array(y)

A = np.zeros((n, degree + 1))
for i in range(degree + 1):
    A[:, i] = (x**i)

coeffs = np.linalg.inv(A.T @ A) @ A.T @ y

for i in range(len(coeffs) - 1, -1, -1):
    print(np.round(coeffs[i], 2), end = ' ')
