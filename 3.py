import numpy as np

# Define the objective function
def f(x):
    x1, x2 = x
    return x1**2 + x2**2 - x1*x2 + x1 - x2

# Define the gradient of the function
def grad_f(x):
    x1, x2 = x
    df_dx1 = 2*x1 - x2 + 1
    df_dx2 = 2*x2 - x1 - 1
    return np.array([df_dx1, df_dx2])

# Wolfe condition line search
def line_search(x, d, grad, alpha=1, c1=1e-4, rho=0.75, epsilon=0.9):
    while f(x + alpha * d) > f(x) + c1 * alpha * np.dot(grad, d):
        alpha *= rho
        if alpha < epsilon:
            break
    return alpha

# Steepest descent method
def steepest_descent(x0, tol=1e-4, max_iter=1000):
    x = np.array(x0, dtype=float)
    for i in range(max_iter):
        grad = grad_f(x)
        if np.linalg.norm(grad) < tol:
            break
        d = -grad  # Steepest descent direction
        alpha = line_search(x, d, grad)  # Line search for step size
        x += alpha * d
    return x, f(x)

# Initial point
x0 = [0, 0]

# Solve the optimization problem
solution, value = steepest_descent(x0)

# Print results
print("Optimal Solution:", solution)
print("Optimal Value:", value)
