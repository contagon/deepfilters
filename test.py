import sympy as sy
import numpy as np

x0, x1 = sy.symbols('x0 x1')
f = sy.Array([x0, x1])

g = sy.lambdify([["x0", "x1"]], f, modules='math')

gcuda = jit(g)
print(gcuda(np.zeros(2)))