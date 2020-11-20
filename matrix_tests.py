'''
Same as vector_tests, but with matrices
dFA/dx and dAF/dx where A is a constant 3x3 matrix

Properties:

Given f(x): k x 1 -> n x 1 and some constant n x l matrix A, it follows:
d(f.T*A)/dx = A.T * df/dx
(This can be checked by writing out dot products)

Then, for F(x): k x 1 -> m x n, with Jacobian defined as
(dF/dx)[i,:,:] = dF[i,:]/dx,
we have
(dFA/dx)[i,:,:] = dFA[i,:]/dx = dF[i,:]A/dx = A.T * dF[i,:]/dx
(2nd equality can also be checked manually)

This is the basis for test2()


For dAF/dx is is more complicated - we need to transpose dF/dx before and after multiplication, so
dAF/dx = (A @ (dF/dx).T).T
0 and 1 axes are transposed both times.
(Can be checked by writing out dAF/dx, and the operations between A and dF/x)

This is similar to vectors, before we had
dvF/dx = (v @ (dF/dx).T).reshape(3,3)
Second transpose was not necessary, since reshaping (and removing axis with size 1) led to the correct matrix.
It might as well be done for the sake of consistency.

'''
import numpy as np
from vector_tests import F, dF

# F dot A
def FA(x, A):
    x1 = x[0, 0]
    x2 = x[1, 0]
    x3 = x[2, 0]
    A1 = A[0, 0]
    A2 = A[0, 1]
    A3 = A[0, 2]
    A4 = A[1, 0]
    A5 = A[1, 1]
    A6 = A[1, 2]
    A7 = A[2, 0]
    A8 = A[2, 1]
    A9 = A[2, 2]
    # Just regular dot product
    FA = np.array([[(x2*x3)*A1 + (3*x1)*A4 + 0, (x2*x3)*A2 + (3*x1)*A5 + 0, (x2*x3)*A3 + (3*x1)*A6 + 0],
                  [(x2**2)*A1 + (5*x3)*A4 + (x1+2*x2+3*x3)*A7, (x2**2)*A2 + (5*x3)*A5 + (x1+2*x2+3*x3)*A8, (x2**2)*A3 + (5*x3)*A6 + (x1+2*x2+3*x3)*A9],
                  [A1 + (x1**2*x2 + x3**3)*A4 + (x2*x3**2)*A7, A2 + (x1**2*x2 + x3**3)*A5 + (x2*x3**2)*A8, A3 + (x1**2*x2 + x3**3)*A6 + (x2*x3**2)*A9]])
    return FA

# dF(x)A/dx
def dFA(x, A):
    x1 = x[0, 0]
    x2 = x[1, 0]
    x3 = x[2, 0]
    A1 = A[0, 0]
    A2 = A[0, 1]
    A3 = A[0, 2]
    A4 = A[1, 0]
    A5 = A[1, 1]
    A6 = A[1, 2]
    A7 = A[2, 0]
    A8 = A[2, 1]
    A9 = A[2, 2]
    # 3x3x3 Jacobian
    dFA = np.array([[[3*A4, x3*A1, x2*A1],
                    [3*A5, x3*A2, x2*A2],
                    [3*A6, x3*A3, x2*A3]],

                   [[A7, 2*x2*A1 + 2*A7, 5*A4 + 3*A7],
                    [A8, 2*x2*A2 + 2*A8, 5*A5 + 3*A8],
                    [A9, 2*x2*A3 + 2*A9, 5*A6 + 3*A9]],

                   [[2*x1*x2*A4,  x1**2*A4 + x3**2*A7,  3*x3**2*A4 + 2*x2*x3*A7],
                    [2*x1*x2*A5,  x1**2*A5 + x3**2*A8,  3*x3**2*A5 + 2*x2*x3*A8],
                    [2*x1*x2*A6,  x1**2*A6 + x3**2*A9,  3*x3**2*A6 + 2*x2*x3*A9]]])
    return dFA


# A dot F
def AF(x, A):
    x1 = x[0, 0]
    x2 = x[1, 0]
    x3 = x[2, 0]
    A1 = A[0, 0]
    A2 = A[0, 1]
    A3 = A[0, 2]
    A4 = A[1, 0]
    A5 = A[1, 1]
    A6 = A[1, 2]
    A7 = A[2, 0]
    A8 = A[2, 1]
    A9 = A[2, 2]
    # Just regular dot product
    AF = np.array([[A1*x2*x3 + A2*x2**2 + A3, 3*A1*x1 + 5*A2*x3 + A3*(x1**2*x2 + x3**3), 0 + (x1 + 2*x2 + 3*x3)*A2 + A3*(x2*x3**2)],
                   [A4*x2*x3 + A5*x2**2 + A6, 3*A4*x1 + 5*A5*x3 + A6*(x1**2*x2 + x3**3), 0 + (x1 + 2*x2 + 3*x3)*A5 + A6*(x2*x3**2)],
                   [A7*x2*x3 + A8*x2**2 + A9, 3*A7*x1 + 5*A8*x3 + A9*(x1**2*x2 + x3**3), 0 + (x1 + 2*x2 + 3*x3)*A8 + A9*(x2*x3**2)]])
    return AF


# dAF(x)/dx
def dAF(x, A):
    x1 = x[0, 0]
    x2 = x[1, 0]
    x3 = x[2, 0]
    A1 = A[0, 0]
    A2 = A[0, 1]
    A3 = A[0, 2]
    A4 = A[1, 0]
    A5 = A[1, 1]
    A6 = A[1, 2]
    A7 = A[2, 0]
    A8 = A[2, 1]
    A9 = A[2, 2]
    # 3x3x3 Jacobian
    dAF = np.array([[[0, A1*x3 + 2*A2*x2, A1*x2],
                    [3*A1 + 2*A3*x1*x2, A3*x1**2, 5*A2 + 3*A3*x3**2],
                    [A2, 2*A2 + A3*x3**2, 3*A2 + 2*A3*x2*x3]],

                   [[0, A4*x3 + 2*A5*x2, A4*x2],
                    [3*A4 + 2*A6*x1*x2, A6*x1**2, 5*A5 + 3*A6*x3**2],
                    [A5, 2*A5 + A6*x3**2, 3*A5 + 2*A6*x2*x3]],

                   [[0, A7*x3+2*A8*x2, A7*x2],
                    [3*A7 + 2*A9*x1*x2, A9*x1**2, 5*A8 + 3*A9*x3**2],
                    [A8, 2*A8 + A9*x3**2, 3*A8 + 2*A9*x2*x3]]])
    return dAF


# Test dFA/dx = dF/dx @ A
def test1():
    x = np.random.randn(3, 1)
    A = np.random.randn(3, 3)
    print(dFA(x, A))
    print((dF(x) @ A))
    assert np.allclose(dFA(x, A), (dF(x) @ A))

# Test dFA/dx = A.T @ dF/dx
def test2():
    x = np.random.randn(3, 1)
    A = np.random.randn(3, 3)
    print(dFA(x, A))
    print(A.T @ dF(x))
    C = np.zeros((3,3,3))
    assert np.allclose(dFA(x, A), (A.T @ dF(x)))

# Test dAF/dx = A @ dF/dx
def test3():
    x = np.random.randn(3, 1)
    A = np.random.randn(3, 3)
    print(dAF(x, A))
    print(A @ dF(x))
    assert np.allclose(dAF(x, A), (A @ dF(x)))

# Test dAF/dx = (A @ (dF/dx).T).T
#Transpose axes 0, 1 before multiplying, then transpose back after multiplying
def test4():
    x = np.random.randn(3, 1)
    A = np.random.randn(3, 3)
    dF_T = dF(x).transpose(1, 0, 2)
    print(dAF(x, A))
    print((A @ dF_T).transpose(1, 0, 2))
    assert np.allclose(dAF(x, A), (A @ dF_T).transpose(1, 0, 2))

def main():
    #test1() #Fails
    test2()
    #test3() #Fails
    test4()


if __name__ == "__main__":
    main()