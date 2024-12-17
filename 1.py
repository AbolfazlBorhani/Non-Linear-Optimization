import numpy as np

def f(x, Q, b, c):
    return 0.5 * x.T @ Q @ x + b.T @ x + c

def gradient(x, Q, b):
    return Q @ x + b

def steepest_descent(Q, b, c, x0, tol=1e-6, max_iter=1000):
    x = x0
    for _ in range(max_iter):
        grad = gradient(x, Q, b)
        if np.linalg.norm(grad) < tol:
            break
        alpha = - (grad.T @ grad) / (grad.T @ Q @ grad)
        x = x + alpha * grad
    return x

# Inputs
Q = np.array([[3, 1], [1, 4]])
b = np.array([-1, 1])
c = 2
x0 = np.array([0, 0])

# Solve
x_opt = steepest_descent(Q, b, c, x0)
result = f(x_opt, Q, b, c)

# Output
print("x*:", x_opt)
print("f(x*):", result)
