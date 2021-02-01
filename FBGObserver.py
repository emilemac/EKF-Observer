import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import pi, pow
import time
from processing import interpolate


def hat(v):
    return np.array([[0.0, -v[2, 0], v[1, 0]],
                     [v[2, 0], 0.0, -v[0, 0]],
                     [-v[1, 0], v[0, 0], 0.0]])


e3 = np.array([0, 0, 1]).reshape(3, 1)


class FBGObserver:

    def __init__(self, G=2.50e+10, E=6.43e+10):
        self.r_0 = np.array([0, 0, 0]).reshape(3, 1)
        self.u_0 = np.array([5., 3., 0.]).reshape(3, 1)  # initial curvature
        self.u_star = np.array([5., 3., 0.]).reshape(3, 1)  # end curvature
        self.r = np.empty((0, 3))
        self.u = np.empty((0, 3))
        self.G = np.empty((0, 9))
        self.s_steps = np.empty((0))

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


    # Given an array of known curvature (e.g. interpolated), find the shape of the CTR
    def shape_from_curvature(self, s, y, u_arr):
        dydt = np.empty([12, 1])
        # first 3 elements of y are r, next 9 are R, next 3 are u
        r = np.array(y[:3]).reshape(3, 1)
        R = np.array([[y[3], y[4], y[5]],
                      [y[6], y[7], y[8]],
                      [y[9], y[10], y[11]]])

        # first get the correct curvature from the array
        idx = int(round(s * (len(u_arr) - 1) / self.length))
        u = u_arr[idx].reshape(3, 1)

        # Derivatives
        dr = R @ e3
        dR = R @ hat(u)

        dydt[:3, :] = dr.reshape(3, 1)
        dydt[3:12, :] = dR.reshape(9, 1)

        assert dydt.shape == (12, 1)
        return dydt.ravel()


    def solve(self, method=0, u_arr=np.empty((0,3))):
        if method == 0:
            y_0 = np.vstack((self.r_0, self.R_0, self.u_0)).ravel()
            s = solve_ivp(lambda s, y: self.ode_eq(s, y), (0, self.length), y_0, method='RK23', max_step=self.step)
        elif method == 1:
            y_0 = np.vstack((self.r_0, self.R_0, self.u_0, self.X_0, self.G_0)).ravel()
            s = solve_ivp(lambda s, y: self.ode_eq_time_vars(s, y), (0, self.length), y_0, method='RK23',
                          max_step=self.step)
        elif method == 2:
            y_0 = np.vstack((self.r_0, self.R_0)).ravel()
            s = solve_ivp(lambda s, y: self.shape_from_curvature(s, y, u_arr), (0, self.length), y_0, method='RK23', max_step=self.step)
        ans = s.y.transpose()
        self.s_steps = s.t.transpose()
        self.r = ans[:, (0, 1, 2)]
        if method != 2:
            self.u = ans[:, (12, 13, 14)]
        if method == 1:
            self.G = ans[:, (24, 25, 26, 27, 28, 29, 30, 31, 32)]


    # Uses curvature observations
    def observer_ode_fbg(self, s, y, observations, locations, interpolate):
        dydt = np.empty([24, 1])
        # first 3 elements of y are r,
        # next 9 are R,
        # next 3 are u,
        # last 9 are P

        r = np.array(y[:3]).reshape(3, 1)
        R = np.array([[y[3], y[4], y[5]],
                      [y[6], y[7], y[8]],
                      [y[9], y[10], y[11]]])
        u = np.array(y[12:15]).reshape(3, 1)
        P = np.array(y[15:]).reshape(3, 3)

        # Derivatives
        e3 = np.array([0, 0, 1]).reshape(3, 1)
        A = self.A_exact(u) # A is du'(s)/du(s), like before (not taking force into account)
        C = np.eye(3) # Since curvature u is observed from FBGs, C = du(s)/du(s) = I

        dP = A @ P + P @ A.T + self.Q - P @ C.T @ np.linalg.inv(self.R) @ C @ P
        H = P @ C.T @ np.linalg.inv(self.R)

        dr = R @ e3
        dR = R @ hat(u)
        h = u
        if interpolate:
            obs_idx = np.argmin(np.abs(s - locations))
            observed_u = observations[obs_idx].reshape(3, 1)  # Observed curvature from FBGs
            du = self.f_u(u, R) + H @ (observed_u - h)  # u' = f(u) + H(y - h(u))
        else:
            min_dist = np.min(np.abs(s - locations))
            if min_dist <= self.length / 100:
                # If sensor location is close enough, then use observed curvature
                min_ix = np.argmin(np.abs(s - locations))
                observed_u = observations[min_ix].reshape(3, 1)  # Observed curvature from FBGs
                du = self.f_u(u, R) + H @ (observed_u - h)  # u' = f(u) + H(y - h(u))
            else:
                # Otherwise only use model
                du = self.f_u(u, R)


        dydt[:3, :] = dr.reshape(3, 1)
        dydt[3:12, :] = dR.reshape(9, 1)
        dydt[12:15, :] = du.reshape(3, 1)
        dydt[15:, :] = dP.reshape(9, 1)

        assert dydt.shape == (24, 1)
        return dydt.ravel()

    def solve_observer(self, observations, locations, method=0, interpolate=True):
        if method == 0:  # Uses FBG curvature information and updates u
            self.R = self.R[0, 0] * np.eye(3)
            y_0 = np.vstack((self.r_0, self.R_0, self.u_0, self.P_0)).ravel()
            s = solve_ivp(lambda s, y: self.observer_ode_fbg(s, y, observations, locations, interpolate), (0, self.length), y_0,
                          method='RK23',
                          max_step=self.step)
        ans = s.y.transpose()
        self.s_points = s.t.transpose()
        self.r = ans[:, (0, 1, 2)]
        self.u = ans[:, (12, 13, 14)]


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
    obs_base = FBGObserver()
    obs_base.solve()
    obs_base.plot(ax, 'Model only', '-r')

    # Simulate observations by perturbing the model
    obs_sim = FBGObserver()
    obs_sim.force = np.array([0.02, 0.01, 0.01]).reshape(3, 1) * 1
    errors = obs_sim.solve_in_time(iterations=50)  # solves recursively in time
    s_steps = obs_sim.s_steps # array of s
    observations = obs_sim.u # array of u
    obs_sim.plot(ax, "Perturbed", '-g')

    '''# Use EKF observer
    obs = FBGObserver()
    # Change Q and R
    obs.Q = np.eye(3) * 1000  # Small Q -> shape will be closer to model, large Q -> closer to observations
    obs.R = np.eye(2) * 1  # Almost the opposite effect of Q
    obs.solve_observer(observations, s_steps, method=0, interpolate=True)
    #obs.plot(ax, 'EKF observer (continuous)', '-b')'''

    # Use only 8 observations of curvature
    k = int(len(s_steps) / 11)
    indices = [k*i for i in range(2,10)]
    # Plot the sensor locations
    ax.scatter(obs_sim.r[indices, 0], obs_sim.r[indices, 1], obs_sim.r[indices, 2], label='Sensor locations')

    # EKF observer + 8 observations
    obs_partial = FBGObserver()
    obs_partial.Q = np.eye(3) * 1000
    obs_partial.R = np.eye(2) * 1
    obs_partial.solve_observer(observations[indices], s_steps[indices], method=0, interpolate=False)
    obs_partial.plot(ax, 'EKF observer from 8 sensors', '-y')

    # Now interpolate those 8 observations
    x, curvatures = interpolate(s_steps[indices], observations[indices], len(s_steps), obs_base.length, method='cubic')
    obs_interp = FBGObserver()
    obs_interp.solve(method=2, u_arr=curvatures)
    obs_interp.plot(ax, 'Interpolated shape', '-m')

    # Interpolation + observer
    obs_interp.Q = np.eye(3) * 1000
    obs_interp.R = np.eye(2) * 1
    obs_interp.solve_observer(curvatures, x, method=0, interpolate=True)
    obs_interp.plot(ax, 'EKF observer + interpolation', '-b')

    plt.grid(True)
    fig.legend()
    plt.show()


if __name__ == "__main__":
    main()
