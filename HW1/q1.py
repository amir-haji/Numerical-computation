from sympy import sympify, diff, lambdify
from sympy.abc import x
from math import factorial, comb
import numpy as np

def divided_diff(x, y):
    '''
    function to calculate the divided
    differences table
    '''
    n = len(y)
    coef = np.zeros([n, n])
    # the first column is y
    coef[:,0] = y
    
    for j in range(1,n):
        for i in range(n-j):
            coef[i][j] = \
           (coef[i+1][j-1] - coef[i][j-1]) / (x[i+j]-x[i])
            
    return coef

def lagrange_coeff(x_data, x):
    coeff = []

    for i in range(len(x_data)):
        ans = 1
        for j in range(len(x_data)):
            if j != i:
                ans *= (x - x_data[j])/(x_data[i]-x_data[j])
        coeff.append(ans)

    return np.array(coeff)

def newton_poly(coef, x_data, x):
    '''
    evaluate the newton polynomial 
    at x
    '''
    n = len(x_data) - 1 
    p = coef[n]
    for k in range(1,n+1):
        p = coef[n-k] + (x -x_data[n-k])*p
    return p


func = sympify(input())
F = lambdify(x, func)
x_data = np.array([float(x) for x in input().split(' ')])
X = float(input())

y = np.array([F(z) for z in x_data])

newton_coef = divided_diff(x_data, y)[0, :]
answer = newton_poly(newton_coef, x_data, X)
lag_coeff = lagrange_coeff(x_data, X)

for i in lag_coeff:
    print(np.round(i, 3), end = ' ')
print()

for  i in newton_coef:
    print(np.round(i, 3), end = ' ')
print()

print(np.round(answer, 3))

