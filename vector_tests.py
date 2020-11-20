'''
Tests np.matmul operator between 3x3x3 tensors and 3x1 vectors
i.e. whether dRe3/du can be written as a product of dR/du and e3 in C'
C = B * dr/du
C' = B * dRe3/du

Given F - n*m matrix, and x - k*1 vector,
dF/dx is n*m*k tensor where

(dF/dx)[i,:,:] = dF[i,:]/dx

a Jacobian of F's i'th row w.r.t x
'''
import numpy as np

# Define function F(x): 3x1 -> 3x3
def F(x):
    x1 = x[0,0]
    x2 = x[1,0]
    x3 = x[2,0]
    F = np.array([[x2*x3,  3*x1,              0],
                  [x2**2,  5*x3,              x1 + 2*x2 + 3*x3],
                  [1,      x1**2*x2 + x3**3,  x2*x3**2]])
    return F


# Analytical solution of dF/dx: 3x3x3
def dF(x):
    x1 = x[0,0]
    x2 = x[1,0]
    x3 = x[2,0]
    dF = np.array([[[0, x3, x2],
                    [3, 0, 0],
                    [0, 0, 0]],

                   [[0, 2*x2, 0],
                    [0, 0, 5],
                    [1, 2, 3]],

                   [[0, 0, 0],
                    [2*x1*x2, x1**2, 3*x3**2],
                    [0, x3**2, 2*x2*x3]]])
    return dF


# Numerical solution of dF/dx
def dF_num(x, e=1e-7):
    dF_num = np.zeros((3,3,3))
    for i in range(3):
        e_i = np.zeros((3,1))
        e_i[i] = e
        xe = x + e_i
        xe_ = x - e_i
        dF_num[:,:,i] = (F(xe) - F(xe_)) / (2*e)
    return dF_num

# F dot v
def Fv(x, v):
    x1 = x[0, 0]
    x2 = x[1, 0]
    x3 = x[2, 0]
    v1 = v[0, 0]
    v2 = v[1, 0]
    v3 = v[2, 0]
    # Just regular dot product
    Fv = np.array([[(x2*x3)*v1 + (3*x1)*v2 + 0],
                  [(x2**2)*v1 + (5*x3)*v2 + (x1+2*x2+3*x3)*v3],
                  [v1 + (x1**2*x2 + x3**3)*v2 + (x2*x3**2)*v3]])
    return Fv

# dF(x)v/dx
def dFv(x, v):
    x1 = x[0, 0]
    x2 = x[1, 0]
    x3 = x[2, 0]
    v1 = v[0, 0]
    v2 = v[1, 0]
    v3 = v[2, 0]
    # Regular 3x3 Jacobian
    dFv = np.array([[3*v2,        x3*v1,                x2*v1],
                    [v3,          2*x2*v1 + 2*v3,       5*v2 + 3*v3],
                    [2*x1*x2*v2,  x1**2*v2 + x3**2*v3,  3*x3**2*v2 + 2*x2*x3*v3]])
    return dFv


# v.T dot F
def vF(x, v):
    x1 = x[0, 0]
    x2 = x[1, 0]
    x3 = x[2, 0]
    v1 = v[0, 0]
    v2 = v[1, 0]
    v3 = v[2, 0]
    # Just regular dot product
    vF = np.array([[v1*x2*x3 + v2*x2**2 + v3],
                  [3*v1*x1 + 5*v2*x3 + v3*(x1**2*x2 + x3**3)],
                  [0 + (x1 + 2*x2 + 3*x3)*v2 + v3*(x2*x3**2)]])
    return vF


# dvF(x)/dx
def dvF(x, v):
    x1 = x[0, 0]
    x2 = x[1, 0]
    x3 = x[2, 0]
    v1 = v[0, 0]
    v2 = v[1, 0]
    v3 = v[2, 0]
    # Regular 3x3 Jacobian
    dvF = np.array([[0,                 v1*x3 + 2*v2*x2,    v1*x2],
                    [3*v1 + 2*v3*x1*x2, v3*x1**2,         5*v2 + 3*v3*x3**2],
                    [v2,                2*v2 + v3*x3**2,  3*v2 + 2*v3*x2*x3]])
    return dvF


# Test if dRe3/du = dR/du @ e3
# General case: dF(x)v/dx = dF(x)/dx @ v, x and v arbitrary
def test1():
    x = np.random.randn(3, 1)
    v = np.random.randn(3, 1)
    print(dFv(x, v))
    print((dF(x) @ v))
    assert np.allclose(dFv(x, v), (dF(x) @ v).reshape(3,3))

# Test dFv/dx = v.T @ dF/dx
def test2():
    x = np.random.randn(3, 1)
    v = np.random.randn(3, 1)
    print(dFv(x, v))
    print((v.T @ dF(x)).reshape(3,3))
    assert np.allclose(dFv(x, v), (v.T @ dF(x)).reshape(3,3))

# Test dFv/dx = (dF/dx).T @ v
# Transpose axes 1 and 2
def test3():
    x = np.random.randn(3, 1)
    v = np.random.randn(3, 1)
    dF_T = dF(x).transpose(0, 2, 1)
    print(dFv(x, v))
    print((dF_T @ v).reshape(3,3))
    assert np.allclose(dFv(x, v), (dF_T @ v).reshape(3,3))

# Test dvF/dx = dF/dx @ v.T
def test4():
    x = np.random.randn(3, 1)
    v = np.random.randn(3, 1)
    print(dvF(x, v))
    print((dF(x) @ v).reshape(3,3))
    assert np.allclose(dvF(x, v), (dF(x) @ v).reshape(3,3))

# Test dvF/dx = v @ (dF/dx).T
# Transpose axes 0 and 1
def test5():
    x = np.random.randn(3, 1)
    v = np.random.randn(3, 1)
    dF_T = dF(x).transpose(1, 0, 2)
    print(dvF(x, v))
    print((v.T @ dF_T).reshape(3,3))
    assert np.allclose(dvF(x, v), ((v.T @ dF_T)).transpose(1,0,2).reshape(3,3))

def main():
    '''x = np.random.randn(3,1)
    print(dF(x) - dF_num(x))'''

    #test1() # Fails
    test2()
    test3()
    #test4() # Fails
    test5()


if __name__ == "__main__":
    main()