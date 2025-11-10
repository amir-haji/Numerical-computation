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
a, b = [float(z) for z in input().split(' ')]
X = float(input())
n = int(input())

k = np.array(list(range(1, n + 1)))
x_uniform = a + (b - a) * (k - 1)/(n - 1)
x_chebyshev = (a + b)/2 + ((b - a) / 2) * np.cos((2 * k - 1) * np.pi/(2 * n))

y_uni = np.array([F(z) for z in x_uniform])
y_che = np.array([F(z) for z in x_chebyshev])

uni_coef = divided_diff(x_uniform, y_uni)[0, :]
uni_answer = newton_poly(uni_coef, x_uniform, X)

che_coef = divided_diff(x_chebyshev, y_che)[0, :]
che_answer = newton_poly(che_coef, x_chebyshev, X)

real_answer = F(X)
diff_error = np.abs(((real_answer - uni_answer) - (real_answer - che_answer)))


print(np.round(che_answer, 3))
print(np.round(uni_answer, 3))
print(np.round(diff_error, 3))