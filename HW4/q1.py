import numpy as np
from numpy.linalg import det
n = int(input())

A = []
b = []

for _ in range(n):
    nums = [float(x) for x in input().split(' ')]
    A.append(nums[:-1])
    b.append(nums[-1])

A = np.array(A)
b = np.array(b)

answers = []

for i in range(n):
    C = A.copy()
    C[:, i] = b.T
    answers.append(det(C) / det(A))

for ans in answers:
    print(np.round(ans, 2), end = ' ')
