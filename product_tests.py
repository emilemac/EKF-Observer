'''
Tests np.matmul operator in the general case of product rule dFG/dx = dF/dx * G + F * dG/dx,
using rules derived in matrix_tests.
dFG/dx = dF/dx * G + F * dG/dx does not work, as test1() fails.
but,
dFG/dx = G.T @ dF/dx + (F @ (dG/dx).T).T
indeed does.

C  = B * dr/du
C' = B * dRe3/du = B * (e3.T @ dR/du)
D  = dR/du
then
D' = dRu^/du
   = (u^).T @ dR/du + (R @ (du^/du).T).T
'''
import numpy as np

# Define function F(x): 3x1 -> 3x3
# Different from vector_tests.py, for simplicity
def F(x):
    x1 = x[0,0]
    x2 = x[1,0]
    x3 = x[2,0]
    F = np.array([[x1, 0, 1],
                  [0, 2, x3],
                  [x2**2, 1, 1]])
    return F


# Analytical solution of dF/dx: 3x3x3
def dF(x):
    x1 = x[0,0]
    x2 = x[1,0]
    x3 = x[2,0]
    dF = np.array([[[1, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]],

                   [[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 1]],

                   [[0, 2*x2, 0],
                    [0, 0, 0],
                    [0, 0, 0]]])
    return dF


# Define function G(x): 3x1 -> 3x3
def G(x):
    x1 = x[0,0]
    x2 = x[1,0]
    x3 = x[2,0]
    G = np.array([[0, x2, x3],
                  [x3, x2, x1],
                  [x1, 0, 0]])
    return G


# Analytical solution of dG/dx: 3x3x3
def dG(x):
    x1 = x[0,0]
    x2 = x[1,0]
    x3 = x[2,0]
    dG = np.array([[[0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]],

                   [[0, 0, 1],
                    [0, 1, 0],
                    [1, 0, 0]],

                   [[1, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]]])
    return dG


# F(x)*G(x): 3x1 -> 3x3
def FG(x):
    x1 = x[0,0]
    x2 = x[1,0]
    x3 = x[2,0]
    FG = np.array([[x1, x1*x2, x1*x3],
                  [2*x3+x1*x3, 2*x2, 2*x1],
                  [x1+x3, x2**3+x2, x2**2*x3+x1]])
    return FG


# Analytical solution of dFG/dx: 3x3x3
def dFG(x):
    x1 = x[0,0]
    x2 = x[1,0]
    x3 = x[2,0]
    dFG = np.array([[[1, 0, 0],
                    [x2, x1, 0],
                    [x3, 0, x1]],

                   [[x3, 0, 2+x1],
                    [0, 2, 0],
                    [2, 0, 0]],

                   [[1, 0, 1],
                    [0, 3*x2**2+1, 0],
                    [1, 2*x2*x3, x2**2]]])
    return dFG


# Test dFG/dx = dF/dx @ G + F @ dG/dx
def test1():
    x = np.random.randn(3, 1)
    product = dF(x) @ G(x) + F(x) @ dG(x)
    print(dFG(x))
    print(product)
    assert np.allclose(dFG(x), product)

# Use rules from matrix_tests
# Test dFG/dx = G.T @ dF/dx + (F @ (dG/dx).T).T
# Transpose axes 0 and 1
def test2():
    x = np.random.randn(3, 1)
    product = G(x).T @ dF(x) + (F(x) @ dG(x).transpose(1,0,2)).transpose(1,0,2)
    print(dFG(x))
    print(product)
    assert np.allclose(dFG(x), product)

def main():
    #test1() #Fails
    test2()

if __name__ == "__main__":
    main()
