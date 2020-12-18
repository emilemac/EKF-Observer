import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import pi, pow
import time


def hat(v):
    return np.array([[0.0, -v[2, 0], v[1, 0]],
                     [v[2, 0], 0.0, -v[0, 0]],
                     [-v[1, 0], v[0, 0], 0.0]])


e3 = np.array([0, 0, 1]).reshape(3, 1)

# du^/du, 3x3x3 array
du_hat_du = np.array([[[0, 0, 0],
                       [0, 0, -1],
                       [0, 1, 0]],

                      [[0, 0, 1],
                       [0, 0, 0],
                       [-1, 0, 0]],

                      [[0, -1, 0],
                       [1, 0, 0],
                       [0, 0, 0]]])


# Given X = du(s)/du(l), finds du^(s)/du(l)
def du_hat(X):
    sum = np.zeros((9, 3))
    for i in range(3):
        e_i = np.zeros((3, 1))
        e_i[i] = 1
        product = np.kron(-hat(e_i) @ X, e_i)  # (-e_i^ * du/du) ⊗ e_i
        sum += product
    return sum.reshape(3, 3, 3)


class Observer:

    def __init__(self, G=2.50e+10, E=6.43e+10):
        self.r_0 = np.array([0, 0, 0]).reshape(3, 1)
        self.u_0 = np.array([5., 3., 0.]).reshape(3, 1)  # initial curvature
        self.u_star = np.array([5., 3., 0.]).reshape(3, 1)  # end curvature
        self.r = np.empty((0, 3))
        self.u = np.empty((0, 3))
        self.G = np.empty((0, 9))

        # Tube parameters from CTR_KinematicModel
        # Joint variables
        self.q = np.array([0.01, 0.015, 0.019, 0, 0, 0])
        # Initial position of joints
        self.q_0 = np.array([-0.2858, -0.2025, -0.0945, 0, 0, 0])
        self.alpha_1_0 = self.q[3] + self.q_0[3]  # initial twist angle for tube 1
        self.R_0 = np.array(
            [[np.cos(self.alpha_1_0), -np.sin(self.alpha_1_0), 0], [np.sin(self.alpha_1_0), np.cos(self.alpha_1_0), 0],
             [0, 0, 1]]).reshape(9, 1)
        self.E = E
        self.G = G
        self.I = (pi * (pow(2 * 0.55e-3, 4) - pow(2 * 0.35e-3, 4))) / 64
        self.J = (pi * (pow(2 * 0.55e-3, 4) - pow(2 * 0.35e-3, 4))) / 32
        self.K = np.diag(np.array([self.E * self.I, self.E * self.I, self.G * self.J]))
        self.length = 431e-3

        self.step = 0.001

        # Initialize Q and R
        self.Q = np.eye(3)
        self.R = np.eye(2)

        # Initial values for the observer
        self.P_0 = np.eye(3).reshape(9, 1)
        self.C_0 = np.zeros((2, 3)).reshape(6, 1)
        self.D_0 = np.zeros((3, 3, 3)).reshape(27, 1)
        self.F_0 = np.zeros((3, 3)).reshape(9, 1)
        self.X_0 = np.zeros((3, 3)).reshape(9, 1)
        self.G_0 = np.eye(3).reshape(9, 1)
        self.Z_0 = np.eye(3).reshape(9, 1)

        self.force = np.array([0., 0., 0.]).reshape(3, 1)

        np.random.seed(0)
        self.B = np.random.randn(3, 3)

    # f(u) = du/ds
    def f_u(self, u, R=np.zeros((3, 3))):
        u = u.reshape(3, 1)
        return -np.linalg.inv(self.K) @ (hat(u) @ self.K @ (u - self.u_star) + hat(e3) @ R.T @ self.force)

    # The exact derivative of f(u) w.r.t. u, ignoring force
    def A_exact(self, u):
        K_uhat = hat(np.dot(self.K, (u - self.u_star)))  # [K(u-u*)]^
        uhat_K = np.dot(hat(u), self.K)  # u^K
        K_inv = np.linalg.inv(self.K)
        A = np.dot(K_inv, (K_uhat - uhat_K))  # K_inv * ([K(u-u*)]^ - u^K)
        return A

    # Ode using only CTR equations
    def ode_eq(self, s, y):
        dydt = np.empty([15, 1])
        # first 3 elements of y are r, next 9 are R, next 3 are u
        r = np.array(y[:3]).reshape(3, 1)
        R = np.array([[y[3], y[4], y[5]],
                      [y[6], y[7], y[8]],
                      [y[9], y[10], y[11]]])
        u = np.array(y[12:]).reshape(3, 1)

        # Derivatives
        e3 = np.array([0, 0, 1]).reshape(3, 1)
        dr = R @ e3
        dR = R @ hat(u)
        du = self.f_u(u, R)

        dydt[:3, :] = dr.reshape(3, 1)
        dydt[3:12, :] = dR.reshape(9, 1)
        dydt[12:15, :] = du.reshape(3, 1)

        assert dydt.shape == (15, 1)
        return dydt.ravel()


    # Ode including time variables X and gamma
    def ode_eq_time_vars(self, s, y):
        dydt = np.empty([33, 1])
        # first 3 elements of y are r, next 9 are R, next 3 are u
        r = np.array(y[:3]).reshape(3, 1)
        R = np.array([[y[3], y[4], y[5]],
                      [y[6], y[7], y[8]],
                      [y[9], y[10], y[11]]])
        u = np.array(y[12:15]).reshape(3, 1)
        X = np.array(y[15:24]).reshape(3, 3)
        G = np.array(y[24:]).reshape(3, 3)

        # Derivatives
        e3 = np.array([0, 0, 1]).reshape(3, 1)
        dr = R @ e3
        dR = R @ hat(u)
        du = -np.linalg.inv(self.K) @ (hat(u) @ self.K @ (u - self.u_star) + hat(e3) @ R.T @ self.force)
        #du = G @ self.u_0
        dX = hat(R.T @ self.force) @ G - hat(u) @ X
        dG = np.linalg.inv(self.K) @ (hat(self.K @ (u - self.u_star)) @ G - hat(u) @ self.K @ G - hat(e3) @ X)

        dydt[:3, :] = dr.reshape(3, 1)
        dydt[3:12, :] = dR.reshape(9, 1)
        dydt[12:15, :] = du.reshape(3, 1)
        dydt[15:24, :] = dX.reshape(9, 1)
        dydt[24:, :] = dG.reshape(9, 1)

        assert dydt.shape == (33, 1)
        return dydt.ravel()

    def solve(self, method=0):
        if method == 0:
            y_0 = np.vstack((self.r_0, self.R_0, self.u_0)).ravel()
            s = solve_ivp(lambda s, y: self.ode_eq(s, y), (0, self.length), y_0, method='RK23', max_step=self.step)
        elif method == 1:
            y_0 = np.vstack((self.r_0, self.R_0, self.u_0, self.X_0, self.G_0)).ravel()
            s = solve_ivp(lambda s, y: self.ode_eq_time_vars(s, y), (0, self.length), y_0, method='RK23',
                          max_step=self.step)
        ans = s.y.transpose()
        self.r = ans[:, (0, 1, 2)]
        self.u = ans[:, (12, 13, 14)]
        if method == 1:
            self.G = ans[:, (24, 25, 26, 27, 28, 29, 30, 31, 32)]


    # Spatial observer ODE where u' is updated using error in r
    # Uses 2D information about r
    def observer_ode_2d_u(self, s, y, observations):
        dydt = np.empty([66, 1])
        # first 3 elements of y are r,
        # next 9 are R,
        # next 3 are u,
        # next 6 are C,
        # next 27 are D,
        # next 27 are Z
        # last 9 are P

        r = np.array(y[:3]).reshape(3, 1)
        R = np.array([[y[3], y[4], y[5]],
                      [y[6], y[7], y[8]],
                      [y[9], y[10], y[11]]])
        u = np.array(y[12:15]).reshape(3, 1)
        C = np.array(y[15:21]).reshape(2, 3)
        D = np.array(y[21:48]).reshape(3, 3, 3)
        Z = np.array(y[48:57]).reshape(3, 3)
        P = np.array(y[57:]).reshape(3, 3)

        # Derivatives
        e3 = np.array([0, 0, 1]).reshape(3, 1)
        A = self.A_exact(u)
        B = np.array([[1, 0, 0], [0, 0, 1]])
        # C = d(Br)/du
        # D = dR/du
        # du^/du is a function of Z
        # Z = du(s)/du(l)
        dC = B @ (e3.T @ D).reshape(3, 3)  # Tensor multiplication equivalent to C' = d(BRe3)/du (see vector_tests.py)
        dD = (hat(u)).T @ D + (R @ (du_hat(Z)).transpose(1, 0, 2)).transpose(1, 0, 2)  # Equivalent to D' = d(Ru^)/du  (see product_tests.py)
        dZ = np.linalg.inv(self.K) @ (hat(self.K @ (u - self.u_0)) @ Z - hat(u) @ self.K @ Z)

        dP = A @ P + P @ A.T + self.Q - P @ C.T @ np.linalg.inv(self.R) @ C @ P
        H = P @ C.T @ np.linalg.inv(self.R)

        dr = R @ e3
        dR = R @ hat(u)
        obs_idx = int(round(s / self.length * (len(observations) - 1)))
        observed_xz = B @ observations[obs_idx].reshape(3, 1) # 2d observations
        h = B @ r
        du = self.f_u(u, R) + H @ (observed_xz - h)  # u' = f(u) + H(y - h(u))

        dydt[:3, :] = dr.reshape(3, 1)
        dydt[3:12, :] = dR.reshape(9, 1)
        dydt[12:15, :] = du.reshape(3, 1)
        dydt[15:21, :] = dC.reshape(6, 1)
        dydt[21:48, :] = dD.reshape(27, 1)
        dydt[48:57, :] = dZ.reshape(9, 1)
        dydt[57:, :] = dP.reshape(9, 1)

        assert dydt.shape == (66, 1)
        return dydt.ravel()

    # Same as observer_ode_2d_u but with time variables X and gamma, so it can be solved in time
    def observer_ode_2d_u_time(self, s, y, observations):
        dydt = np.empty([84, 1])
        # first 3 elements of y are r,
        # next 9 are R,
        # next 3 are u,
        # next 6 are C,
        # next 27 are D,
        # next 27 are Z
        # last 9 are P

        r = np.array(y[:3]).reshape(3, 1)
        R = np.array([[y[3], y[4], y[5]],
                      [y[6], y[7], y[8]],
                      [y[9], y[10], y[11]]])
        u = np.array(y[12:15]).reshape(3, 1)
        C = np.array(y[15:21]).reshape(2, 3)
        D = np.array(y[21:48]).reshape(3, 3, 3)
        Z = np.array(y[48:57]).reshape(3, 3)
        P = np.array(y[57:66]).reshape(3, 3)
        X = np.array(y[66:75]).reshape(3, 3)
        G = np.array(y[75:]).reshape(3, 3)

        # Derivatives
        e3 = np.array([0, 0, 1]).reshape(3, 1)
        A = self.A_exact(u)
        B = np.array([[1, 0, 0], [0, 0, 1]])
        # C = d(Br)/du
        # D = dR/du
        # du^/du is a function of Z
        # Z = du(s)/du(l)
        dC = B @ (e3.T @ D).reshape(3,3)  # Tensor multiplication equivalent to C' = d(BRe3)/du (see vector_tests.py)
        dD = (hat(u)).T @ D + (R @ (du_hat(Z)).transpose(1, 0, 2)).transpose(1, 0, 2)  # Equivalent to D' = d(Ru^)/du  (see product_tests.py)
        dZ = np.linalg.inv(self.K) @ (hat(self.K @ (u - self.u_0)) @ Z - hat(u) @ self.K @ Z)

        dP = A @ P + P @ A.T + self.Q - P @ C.T @ np.linalg.inv(self.R) @ C @ P
        H = P @ C.T @ np.linalg.inv(self.R)

        dr = R @ e3
        dR = R @ hat(u)
        obs_idx = int(round(s / self.length * (len(observations) - 1)))
        observed_xz = B @ observations[obs_idx].reshape(3, 1) # 2d observations
        h = B @ r
        du = self.f_u(u, R) + H @ (observed_xz - h)  # u' = f(u) + H(y - h(u))

        dX = hat(R.T @ self.force) @ G - hat(u) @ X
        dG = np.linalg.inv(self.K) @ (hat(self.K @ (u - self.u_star)) @ G - hat(u) @ self.K @ G - hat(e3) @ X)

        dydt[:3, :] = dr.reshape(3, 1)
        dydt[3:12, :] = dR.reshape(9, 1)
        dydt[12:15, :] = du.reshape(3, 1)
        dydt[15:21, :] = dC.reshape(6, 1)
        dydt[21:48, :] = dD.reshape(27, 1)
        dydt[48:57, :] = dZ.reshape(9, 1)
        dydt[57:66, :] = dP.reshape(9, 1)
        dydt[66:75, :] = dX.reshape(9, 1)
        dydt[75:, :] = dG.reshape(9, 1)

        assert dydt.shape == (84, 1)
        return dydt.ravel()


    # Same as observer_ode_2d_u, but uses 3D information about observed r
    def observer_ode_3d_u(self, s, y, observations):
        dydt = np.empty([69, 1])
        # first 3 elements of y are r,
        # next 9 are R,
        # next 3 are u,
        # next 9 are C,
        # next 27 are D,
        # next 27 are Z
        # last 9 are P

        r = np.array(y[:3]).reshape(3, 1)
        R = np.array([[y[3], y[4], y[5]],
                      [y[6], y[7], y[8]],
                      [y[9], y[10], y[11]]])
        u = np.array(y[12:15]).reshape(3, 1)
        C = np.array(y[15:24]).reshape(3, 3)
        D = np.array(y[24:51]).reshape(3, 3, 3)
        Z = np.array(y[51:60]).reshape(3, 3)
        P = np.array(y[60:]).reshape(3, 3)

        # Derivatives
        e3 = np.array([0, 0, 1]).reshape(3, 1)
        A = self.A_exact(u)
        B = np.eye(3)
        # C = d(Br)/du
        # D = dR/du
        # du^/du is a function of Z
        # Z = du(s)/du(l)
        dC = B @ (e3.T @ D).reshape(3, 3)  # Tensor multiplication equivalent to C' = d(BRe3)/du (see vector_tests.py)
        dD = (hat(u)).T @ D + (R @ (du_hat(Z)).transpose(1, 0, 2)).transpose(1, 0, 2)  # Equivalent to D' = d(Ru^)/du  (see product_tests.py)
        dZ = np.linalg.inv(self.K) @ (hat(self.K @ (u - self.u_0)) @ Z - hat(u) @ self.K @ Z)

        dP = A @ P + P @ A.T + self.Q - P @ C.T @ np.linalg.inv(self.R) @ C @ P
        H = P @ C.T @ np.linalg.inv(self.R)

        dr = R @ e3
        dR = R @ hat(u)
        obs_idx = int(round(s / self.length * (len(observations) - 1)))
        observed_xz = B @ observations[obs_idx].reshape(3, 1) # 3d observations
        h = B @ r
        du = self.f_u(u, R) + H @ (observed_xz - h)  # u' = f(u) + H(y - h(u))

        dydt[:3, :] = dr.reshape(3, 1)
        dydt[3:12, :] = dR.reshape(9, 1)
        dydt[12:15, :] = du.reshape(3, 1)
        dydt[15:24, :] = dC.reshape(9, 1)
        dydt[24:51, :] = dD.reshape(27, 1)
        dydt[51:60, :] = dZ.reshape(9, 1)
        dydt[60:, :] = dP.reshape(9, 1)

        assert dydt.shape == (69, 1)
        return dydt.ravel()


    # Updates r' instead of u'
    # A = dr'(s)/dr(l), C = B
    def observer_ode_3d_r(self, s, y, observations):
        dydt = np.empty([60, 1])
        # first 3 elements of y are r,
        # next 9 are R,
        # next 3 are u,
        # next 27 are D,
        # next 9 are F
        # last 9 are P

        r = np.array(y[:3]).reshape(3, 1)
        R = np.array([[y[3], y[4], y[5]],
                      [y[6], y[7], y[8]],
                      [y[9], y[10], y[11]]])
        u = np.array(y[12:15]).reshape(3, 1)
        D = np.array(y[15:42]).reshape(3, 3, 3)
        F = np.array(y[42:51]).reshape(3, 3)
        P = np.array(y[51:60]).reshape(3, 3)

        e3 = np.array([0, 0, 1]).reshape(3, 1)
        A = (e3.T @ D).reshape(3, 3)  # Actually this is always 0
        B = np.eye(3)
        C = B
        E = F.T @ du_hat_du
        # Derivatives
        dD = (hat(u)).T @ D + (R @ (E).transpose(1, 0, 2)).transpose(1, 0, 2)
        dF = np.linalg.inv(self.K) @ (hat(self.K @ (u - self.u_0)) @ F - hat(u) @ self.K @ F)

        dP = A @ P + P @ A.T + self.Q - P @ C.T @ np.linalg.inv(self.R) @ C @ P
        H = P @ C.T @ np.linalg.inv(self.R)

        obs_idx = int(round(s / self.length * (len(observations) - 1)))
        observed_xyz = observations[obs_idx].reshape(3, 1)
        h = B @ r
        dr = R @ e3 + H @ (B @ observed_xyz - h) # Update step
        dR = R @ hat(u)
        du = self.f_u(u, R)

        dydt[:3, :] = dr.reshape(3, 1)
        dydt[3:12, :] = dR.reshape(9, 1)
        dydt[12:15, :] = du.reshape(3, 1)
        dydt[15:42, :] = dD.reshape(27, 1)
        dydt[42:51, :] = dF.reshape(9, 1)
        dydt[51:60, :] = dP.reshape(9, 1)

        assert dydt.shape == (60, 1)
        return dydt.ravel()


    # Same as observer_ode_3d_new_r, but uses 2d observations
    def observer_ode_2d_r(self, s, y, observations):
        dydt = np.empty([60, 1])
        # first 3 elements of y are r,
        # next 9 are R,
        # next 3 are u,
        # next 27 are D,
        # next 9 are F
        # last 9 are P

        r = np.array(y[:3]).reshape(3, 1)
        R = np.array([[y[3], y[4], y[5]],
                      [y[6], y[7], y[8]],
                      [y[9], y[10], y[11]]])
        u = np.array(y[12:15]).reshape(3, 1)
        D = np.array(y[15:42]).reshape(3, 3, 3)
        F = np.array(y[42:51]).reshape(3, 3)
        P = np.array(y[51:]).reshape(3, 3)

        A = (e3.T @ D).reshape(3, 3)  # Actually this is always 0
        #B = np.array([[0.5, 0, 0.5], [0, 0.5, 0.5]])
        B = np.array([[1, 0, 0], [0, 0, 1]])
        C = B
        E = F.T @ du_hat_du
        # Derivatives
        dD = (hat(u)).T @ D + (R @ (E).transpose(1, 0, 2)).transpose(1, 0, 2)
        dF = np.linalg.inv(self.K) @ (
                hat(self.K @ (u - self.u_0)) @ F - hat(u) @ self.K @ F - hat(e3) @ (self.force.T @ D).reshape(3, 3))

        dP = A @ P + P @ A.T + self.Q - P @ C.T @ np.linalg.inv(self.R) @ C @ P
        H = P @ C.T @ np.linalg.inv(self.R)

        obs_idx = int(round(s / self.length * (len(observations) - 1)))
        observed_xz = B @ observations[obs_idx].reshape(3, 1)
        h = B @ r
        dr = R @ e3 + H @ (observed_xz - h) # Update step
        dR = R @ hat(u)
        du = self.f_u(u, R)

        dydt[:3, :] = dr.reshape(3, 1)
        dydt[3:12, :] = dR.reshape(9, 1)
        dydt[12:15, :] = du.reshape(3, 1)
        dydt[15:42, :] = dD.reshape(27, 1)
        dydt[42:51, :] = dF.reshape(9, 1)
        dydt[51:, :] = dP.reshape(9, 1)

        assert dydt.shape == (60, 1)
        return dydt.ravel()

    # Same as observer_ode_2d_r but with time variables X and gamma, so it can be solved in time
    def observer_ode_2d_r_time(self, s, y, observations):
        dydt = np.empty([78, 1])
        # first 3 elements of y are r,
        # next 9 are R,
        # next 3 are u,
        # next 27 are D,
        # next 9 are F
        # last 9 are P

        r = np.array(y[:3]).reshape(3, 1)
        R = np.array([[y[3], y[4], y[5]],
                      [y[6], y[7], y[8]],
                      [y[9], y[10], y[11]]])
        u = np.array(y[12:15]).reshape(3, 1)
        D = np.array(y[15:42]).reshape(3, 3, 3)
        F = np.array(y[42:51]).reshape(3, 3)
        P = np.array(y[51:60]).reshape(3, 3)
        X = np.array(y[60:69]).reshape(3, 3)
        G = np.array(y[69:]).reshape(3, 3)

        A = (e3.T @ D).reshape(3, 3)  # Always 0
        B = np.array([[1, 0, 0], [0, 0, 1]])
        C = B
        E = F.T @ du_hat_du
        # Derivatives
        dD = (hat(u)).T @ D + (R @ (E).transpose(1, 0, 2)).transpose(1, 0, 2)
        dF = np.linalg.inv(self.K) @ (
                hat(self.K @ (u - self.u_0)) @ F - hat(u) @ self.K @ F - hat(e3) @ (self.force.T @ D).reshape(3, 3))

        dP = A @ P + P @ A.T + self.Q - P @ C.T @ np.linalg.inv(self.R) @ C @ P
        H = P @ C.T @ np.linalg.inv(self.R)

        obs_idx = int(round(s / self.length * (len(observations) - 1)))
        observed_xz = B @ observations[obs_idx].reshape(3, 1)
        h = B @ r
        dr = R @ e3 + H @ (observed_xz - h) # Update step
        dR = R @ hat(u)
        du = self.f_u(u, R)

        # X and gamma derivatives
        dX = hat(R.T @ self.force) @ G - hat(u) @ X
        dG = np.linalg.inv(self.K) @ (hat(self.K @ (u - self.u_star)) @ G - hat(u) @ self.K @ G - hat(e3) @ X)

        dydt[:3, :] = dr.reshape(3, 1)
        dydt[3:12, :] = dR.reshape(9, 1)
        dydt[12:15, :] = du.reshape(3, 1)
        dydt[15:42, :] = dD.reshape(27, 1)
        dydt[42:51, :] = dF.reshape(9, 1)
        dydt[51:60, :] = dP.reshape(9, 1)
        dydt[60:69, :] = dX.reshape(9, 1)
        dydt[69:, :] = dG.reshape(9, 1)

        assert dydt.shape == (78, 1)
        return dydt.ravel()

    def solve_observer(self, observations, method=0):
        if method == 0:  # Uses 3D information and updates r
            self.R = self.R[0, 0] * np.eye(3)
            y_0 = np.vstack((self.r_0, self.R_0, self.u_0, self.D_0, self.F_0, self.P_0)).ravel()
            s = solve_ivp(lambda s, y: self.observer_ode_3d_r(s, y, observations), (0, self.length), y_0,
                          method='RK23',
                          max_step=self.step)
        elif method == 1:  # Uses 2D information and updates r
            y_0 = np.vstack((self.r_0, self.R_0, self.u_0, self.D_0, self.F_0, self.P_0)).ravel()
            s = solve_ivp(lambda s, y: self.observer_ode_2d_r(s, y, observations), (0, self.length), y_0,
                          method='RK23',
                          max_step=self.step)
        elif method == 2:  # Uses 2D information and updates u
            y_0 = np.vstack((self.r_0, self.R_0, self.u_0, self.C_0, self.D_0, self.Z_0, self.P_0)).ravel()
            s = solve_ivp(lambda s, y: self.observer_ode_2d_u(s, y, observations), (0, self.length), y_0,
                          method='RK23',
                          max_step=self.step)
        elif method == 3:  # Uses 3D information and updates u
            self.C_0 = np.zeros((3, 3)).reshape(9, 1)
            self.R = self.R[0, 0] * np.eye(3)
            y_0 = np.vstack((self.r_0, self.R_0, self.u_0, self.C_0, self.D_0, self.Z_0, self.P_0)).ravel()
            s = solve_ivp(lambda s, y: self.observer_ode_3d_u(s, y, observations), (0, self.length), y_0,
                          method='RK23',
                          max_step=self.step)
        elif method == 4:  # Uses 2D information and updates r, also solves X and gamma (time vars)
            y_0 = np.vstack((self.r_0, self.R_0, self.u_0, self.D_0, self.F_0, self.P_0, self.X_0, self.G_0)).ravel()
            s = solve_ivp(lambda s, y: self.observer_ode_2d_r_time(s, y, observations), (0, self.length), y_0,
                          method='RK23',
                          max_step=self.step)
        elif method == 5:  # Uses 2D information and updates u, also solves X and gamma (time vars)
            y_0 = np.vstack((self.r_0, self.R_0, self.u_0, self.C_0, self.D_0, self.Z_0, self.P_0, self.X_0, self.G_0)).ravel()
            s = solve_ivp(lambda s, y: self.observer_ode_2d_u_time(s, y, observations), (0, self.length), y_0,
                          method='RK23',
                          max_step=self.step)
        ans = s.y.transpose()
        self.r = ans[:, (0, 1, 2)]
        self.u = ans[:, (12, 13, 14)]
        if method == 4:
            self.G = ans[:, (69, 70, 71, 72, 73, 74, 75, 76, 77)]
        if method == 5:
            self.G = ans[:, (75, 76, 77, 78, 79, 80, 81, 82, 83)]

    def plot(self, ax, title, color):
        # plot the robot shape
        ax.plot(self.r[:, 0], self.r[:, 1], self.r[:, 2], color, label=title)
        ax.auto_scale_xyz([np.amin(self.r[:, 0]), np.amax(self.r[:, 0]) + 0.01],
                          [np.amin(self.r[:, 1]), np.amax(self.r[:, 1]) + 0.01],
                          [np.amin(self.r[:, 2]), np.amax(self.r[:, 2]) + 0.01])
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_zlabel('Z [mm]')


    def solve_in_time(self, iterations=100, step=0.001, observer=False, observations=None, method = 4):
        P = np.eye(3)
        Q = np.eye(3) * 30
        R = np.eye(3) * 120
        u_0 = self.u_0 #np.array([0., 0., 0.]).reshape(3, 1)
        errors = []
        for t in range(0, iterations):
            T = t * step  # time

            # Solve the equations in the spatial domain:
            self.u_0 = u_0  # set initial u
            if observer:
                # If using spatial observer, need observations
                assert observations is not None
                self.solve_observer(observations, method=method)
            else:
                self.solve(method=1)

            # Get gamma and u of the endpoint
            G = self.G[-1].reshape(3, 3)
            u = self.u[-1].reshape(3, 1)

            # Update time-dependent variables P and u_0
            dP = -P @ G.T @ R @ G @ P + Q
            P = P + step * dP

            error = u - self.u_star
            du_0 = - R @ G.T @ P @ error
            u_0 = u_0 + step * du_0

            errors.append(np.linalg.norm(error))

        return errors


def main():
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # Base - solves r', R', and u', with initial u_0 = u*, no force
    obs_base = Observer()
    #errors1 = obs_base.solve_in_time() - uncomment to solve recursively in time, although it gives the same result (error = 0 at all t)
    tic = time.time()
    obs_base.solve()
    toc = time.time()
    print('Simple ODE: ', toc-tic, 's')
    obs_base.plot(ax, 'Model only', '-r')

    # Simulate observations by perturbing the model with added force
    obs_sim = Observer()
    obs_sim.force = np.array([0.01, 0.01, 0.01]).reshape(3, 1) * 1
    tic = time.time()
    #obs_sim.solve() - uncomment to solve only once spatially, although u_0 is not going to be correct
    errors2 = obs_sim.solve_in_time(iterations=100) # solves recursively in time
    toc = time.time()
    print('100 iterations: ', toc-tic, 's')
    observations = obs_sim.r
    obs_sim.plot(ax, "Observations at T=0.1 (100 iterations)", '-g')

    # Use EKF observer
    obs = Observer()
    # Change Q and R
    obs.Q = np.eye(3) * 1000000  # Small Q -> shape will be closer to model, large Q -> closer to observations
    obs.R = np.eye(2) * 1  # Almost the opposite effect of Q
    tic = time.time()
    obs.solve_observer(observations, method=2)  # method=1 will update r' instead of u'
    toc = time.time()
    print('Observer ODE: ', toc-tic, 's')
    obs.plot(ax, 'EKF observer at T=0', '-b')
    # Solve over time
    tic = time.time()
    errors3 = obs.solve_in_time(iterations=100, observer=True, observations = observations, method=5)  # method=4 will update r' instead of u'
    toc = time.time()
    print('100 iterations: ', toc-tic, 's')
    obs.plot(ax, 'EKF observer at T=0.1 (100 iterations)', '-y')

    fig1 = plt.figure()
    ax1 = plt.axes()
    ax1.plot(errors2, '-b')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Error in u (observations)')
    plt.grid(True)

    fig2 = plt.figure()
    ax2 = plt.axes()
    ax2.plot(errors3, '-b')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Error in u (observer)')

    plt.grid(True)
    fig.legend()
    plt.show()


if __name__ == "__main__":
    main()
