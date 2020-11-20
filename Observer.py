import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import pi, pow


def hat(v):
    return np.array([[0.0, -v[2, 0], v[1, 0]],
                     [v[2, 0], 0.0, -v[0, 0]],
                     [-v[1, 0], v[0, 0], 0.0]])


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


class Observer:

    def __init__(self, G=2.50e+10, E=6.43e+10):
        self.r_0 = np.array([0, 0, 0]).reshape(3, 1)
        self.u_0 = np.array([5, 3, 3]).reshape(3, 1)  # initial curvature
        self.r = np.empty((0, 3))
        self.u = np.empty((0, 3))

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

        self.u_star = np.array([14, 5, 0]).reshape(3, 1)
        self.step = 0.001

        # Initial values for the observer
        self.Q = np.eye(3) * 1000
        self.R = np.eye(2) * 0.001
        self.P_0 = np.eye(3).reshape(9, 1) * 1
        self.C_0 = np.zeros((2, 3)).reshape(6, 1)
        self.D_0 = np.zeros((3, 3, 3)).reshape(27, 1)


    # f(u) = du/ds
    def f_u(self, u):
        u = u.reshape(3, 1)
        return -np.linalg.inv(self.K) @ (hat(u) @ self.K @ (u - self.u_star))


    # Approximation of A: Aij = (f(u+Δuj) - f(u-Δuj))i / 2Δui
    # (adding and subtracting Δu on both sides gives more accurate values around u)
    def A_approx(self, u):
        step = 0.0001
        du0 = np.array([step, 0, 0]).reshape(3, 1)
        du1 = np.array([0, step, 0]).reshape(3, 1)
        du2 = np.array([0, 0, step]).reshape(3, 1)
        df0 = self.f_u(u + du0) - self.f_u(u - du0)
        df1 = self.f_u(u + du1) - self.f_u(u - du1)
        df2 = self.f_u(u + du2) - self.f_u(u - du2)
        df = np.concatenate((df0, df1, df2), axis=1)
        A = df / (2 * step)
        return A


    # The exact derivative of f(u) w.r.t. u, ignoring force
    def A_exact(self, u):
        K_uhat = hat(np.dot(self.K, (u - self.u_star)))  # [K(u-u*)]^
        uhat_K = np.dot(hat(u), self.K)  # u^K
        K_inv = np.linalg.inv(self.K)
        A = np.dot(K_inv, (K_uhat - uhat_K))  # K_inv * ([K(u-u*)]^ - u^K)
        return A


    # Ode using only CTR equations
    def ode_eq(self, s, y, method=0):
        dydt = np.empty([15, 1])
        # first 3 elements of y are r, next 9 are R, next 3 are u
        r = np.array(y[:3]).reshape(3, 1)
        R = np.array([[y[3], y[4], y[5]],
                      [y[6], y[7], y[8]],
                      [y[9], y[10], y[11]]])
        u = np.array(y[12:]).reshape(3, 1)
        # A = np.array([[y[15], y[16], y[17]],[y[18], y[19], y[20]], [y[21], y[22], y[23]]])

        # Derivatives
        e3 = np.array([0, 0, 1]).reshape(3, 1)
        dr = R @ e3
        dR = R @ hat(u)

        if method == 0:
            du = self.f_u(u)
        elif method == 1:
            du = np.dot(self.A_exact(u), u)  # A = df/du -> f = A*u + C, f= u'
        else:
            du = np.dot(self.A_approx(u), u)

        dydt[:3, :] = dr.reshape(3, 1)
        dydt[3:12, :] = dR.reshape(9, 1)
        dydt[12:15, :] = du.reshape(3, 1)

        assert dydt.shape == (15, 1)
        return dydt.ravel()


    def solve(self, method):
        y_0 = np.vstack((self.r_0, self.R_0, self.u_0)).ravel()
        s = solve_ivp(lambda s, y: self.ode_eq(s, y, method), (0, self.length), y_0, method='RK23', max_step=self.step)
        ans = s.y.transpose()
        self.r = np.vstack((self.r, ans[:, (0, 1, 2)]))
        self.u = np.vstack((self.r, ans[:, (12, 13, 14)]))


    def observer_ode_tensors(self, s, y, observations):
        dydt = np.empty([57, 1])
        # first 3 elements of y are r,
        # next 9 are R,
        # next 3 are u,
        # next 6 are C,
        # next 27 are D
        # last 9 are P

        r = np.array(y[:3]).reshape(3, 1)
        R = np.array([[y[3], y[4], y[5]],
                      [y[6], y[7], y[8]],
                      [y[9], y[10], y[11]]])
        u = np.array(y[12:15]).reshape(3, 1)
        C = np.array(y[15:21]).reshape(2, 3)
        D = np.array(y[21:48]).reshape(3, 3, 3)
        P = np.array(y[48:]).reshape(3, 3)

        # Derivatives
        e3 = np.array([0, 0, 1]).reshape(3, 1)
        A = self.A_exact(u)
        B = np.array([[1, 0, 0], [0, 0, 1]])
        #dC = B @ D @ e3
        #dD = D @ hat(u) + R @ du_hat_du
        dC = B @ (e3.T @ D).reshape(3,3)
        dD = (hat(u)).T @ D + (R @ (du_hat_du).transpose(1,0,2)).transpose(1,0,2)

        dP = A @ P + P @ A.T + self.Q - P @ C.T @ np.linalg.inv(self.R) @ C @ P
        H = P @ C.T @ np.linalg.inv(self.R)

        dr = R @ e3
        dR = R @ hat(u)
        obs_idx = int(round(s / self.length * (len(observations) - 1)))
        observed_xz = observations[obs_idx].reshape(2, 1)
        h = B @ r
        du = self.f_u(u) + H @ (observed_xz - h)  # u' = f(u) + H(y - h(u))

        dydt[:3, :] = dr.reshape(3, 1)
        dydt[3:12, :] = dR.reshape(9, 1)
        dydt[12:15, :] = du.reshape(3, 1)
        dydt[15:21, :] = dC.reshape(6, 1)
        dydt[21:48, :] = dD.reshape(27, 1)
        dydt[48:, :] = dP.reshape(9, 1)

        assert dydt.shape == (57, 1)
        return dydt.ravel()


    def solve_observer(self, observations, method=0):
        if method == 0:
            self.C_0 = np.zeros((2, 3)).reshape(6, 1)
            self.D_0 = np.zeros((3, 3, 3)).reshape(27, 1)
            y_0 = np.vstack((self.r_0, self.R_0, self.u_0, self.C_0, self.D_0, self.P_0)).ravel()
            s = solve_ivp(lambda s, y: self.observer_ode_tensors(s, y, observations), (0, self.length), y_0, method='RK23',
                          max_step=self.step)

        ans = s.y.transpose()
        self.r = np.vstack((self.r, ans[:, (0, 1, 2)]))
        self.u = np.vstack((self.u, ans[:, (12, 13, 14)]))


    def plot(self, ax, title, color):
        # plot the robot shape
        ax.plot(self.r[:, 0], self.r[:, 1], self.r[:, 2], color, label=title)
        ax.auto_scale_xyz([np.amin(self.r[:, 0]), np.amax(self.r[:, 0]) + 0.01],
                          [np.amin(self.r[:, 1]), np.amax(self.r[:, 1]) + 0.01],
                          [np.amin(self.r[:, 2]), np.amax(self.r[:, 2]) + 0.01])
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_zlabel('Z [mm]')


def main():

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # Simulate observations
    obs_sim = Observer(G=2e+10, E=7e+10)  # perturb the model
    obs_sim.solve(0)
    observations = obs_sim.r[:, [0, 2]]  # Only have x and z values
    obs_sim.plot(ax, "Observations", '-g')

    obs_base = Observer()
    obs_base.solve(0)
    obs_base.plot(ax, 'Model only', '-r')

    obs = Observer()
    obs.solve_observer(observations, method=0)
    obs.plot(ax, 'EKF observer', '-b')
    print(obs.u[-1] - obs.u_star.T)

    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
