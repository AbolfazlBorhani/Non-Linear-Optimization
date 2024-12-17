import numpy as np
from scipy.linalg import solve

# Objective Function
def f(x):
    x1, x2, x3 = x
    return x1**4 + x2**2 + x3**2 + 2*x1*x2 + x1*x3 - x1*x2 + x1 - x2

# Gradient of the Objective Function
def grad_f(x):
    x1, x2, x3 = x
    df1 = 4*x1**3 + 2*x2 + x3 - x2 + 1
    df2 = 2*x2 + 2*x1 - x1 - 1
    df3 = 2*x3 + x1
    return np.array([df1, df2, df3])

# Hessian of the Objective Function
def hessian_f(x):
    x1, _, _ = x
    h11 = 12*x1**2
    h12 = 2
    h13 = 1
    h21 = 2
    h22 = 2
    h23 = 0
    h31 = 1
    h32 = 0
    h33 = 2
    return np.array([[h11, h12, h13], [h21, h22, h23], [h31, h32, h33]])

# Quadratic Function
Q = np.array([[10, 1, 1], [1, 2, -1], [1, -1, 6]])
def f_quadratic(x):
    return 0.5 * np.dot(x.T, np.dot(Q, x))

def grad_f_quadratic(x):
    return np.dot(Q, x)

def hessian_f_quadratic():
    return Q

# Newton's Method
def newton_method(f_grad, f_hessian, x0, tol=1e-6, max_iter=100):
    x = x0
    for _ in range(max_iter):
        g = f_grad(x)
        H = f_hessian(x)
        delta_x = solve(H, -g)
        x = x + delta_x
        if np.linalg.norm(delta_x) < tol:
            break
    return x

# Steepest Descent Method
def steepest_descent(f_grad, x0, tol=1e-6, max_iter=100, alpha=0.01):
    x = x0
    for _ in range(max_iter):
        g = f_grad(x)
        x = x - alpha * g
        if np.linalg.norm(g) < tol:
            break
    return x

# Combined Method: Steepest Descent followed by Newton's Method
def combined_method(f_grad, f_hessian, x0, tol=1e-6, max_iter=100):
    x = steepest_descent(f_grad, x0, tol, max_iter // 2)
    x = newton_method(f_grad, f_hessian, x, tol, max_iter // 2)
    return x

# Newton's Method with Step Size
def newton_with_step_size(f_grad, f_hessian, x0, tol=1e-6, max_iter=100, alpha=0.5):
    x = x0
    for _ in range(max_iter):
        g = f_grad(x)
        H = f_hessian(x)
        delta_x = solve(H, -g)
        x = x + alpha * delta_x
        if np.linalg.norm(delta_x) < tol:
            break
    return x

# Initial Point
x0 = np.array([0.5, 0.5, 0.5])

# Solving for each method
x_newton = newton_method(grad_f, hessian_f, x0)
x_steepest = steepest_descent(grad_f, x0)
x_combined = combined_method(grad_f, hessian_f, x0)
x_newton_step = newton_with_step_size(grad_f, hessian_f, x0)

# Printing Results
print("Newton's Method Solution:", x_newton)
print("Steepest Descent Solution:", x_steepest)
print("Combined Method Solution:", x_combined)
print("Newton's Method with Step Size Solution:", x_newton_step)

