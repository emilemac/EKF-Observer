import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import pi, pow

def hat(v):
    return np.array([[0,-v[2],v[1]],
                      [v[2],0,-v[0]],
                      [-v[1],v[0],0]])


class Observer:

    def __init__(self):
        self.r_0 = np.array([0, 0, 0]).reshape(3, 1)
        self.u_0 = np.array([6, 3, 0]).reshape(3, 1)  # initial curvature
        self.r = np.empty((0, 3))

        # Tube parameters reused from CTR_KinematicModel
        # Joint variables
        self.q = np.array([0.01, 0.015, 0.019, np.pi / 2, 5 * np.pi / 2, 3 * np.pi / 2])
        # Initial position of joints
        self.q_0 = np.array([-0.2858, -0.2025, -0.0945, 0, 0, 0])
        self.alpha_1_0 = self.q[3] + self.q_0[3]  # initial twist angle for tube 1
        self.R_0 = np.array(
            [[np.cos(self.alpha_1_0), -np.sin(self.alpha_1_0), 0], [np.sin(self.alpha_1_0), np.cos(self.alpha_1_0), 0],
             [0, 0, 1]]).reshape(9, 1)
        self.E = 6.43e+10
        self.G = 2.50e+10
        self.I = (pi * (pow(2 * 0.55e-3, 4) - pow(2 * 0.35e-3, 4))) / 64
        self.J = (pi * (pow(2 * 0.55e-3, 4) - pow(2 * 0.35e-3, 4))) / 32
        self.K = np.diag(np.array([self.E*self.I,self.E*self.I,self.G*self.J]))
        self.length = 431e-3


        self.u_star = np.array([14,5,0]).T
        self.step = 0.001


    # f(u) = du/ds
    def f_u(self, u):
        return -np.dot(np.linalg.inv(self.K), (np.dot(np.dot(hat(u), self.K),(u - self.u_star))))

    # Approximation of A: Aij = (f(u+Δuj) - f(u))i / Δui
    def A_approx(self, u):
        step = 0.001
        du0 = np.array([step, 0, 0]).T
        du1 = np.array([0, step, 0]).T
        du2 = np.array([0, 0, step]).T
        df0 = self.f_u(u + du0) - self.f_u(u)
        df1 = self.f_u(u + du1) - self.f_u(u)
        df2 = self.f_u(u + du2) - self.f_u(u)
        df = np.array([df0,df1,df2])
        A = df / step
        return A

    # The exact derivative of f(u) w.r.t. u, ignoring force
    def A_exact(self, u):
        K_uhat = hat(np.dot(self.K,(u-self.u_star)))    # [K(u-u*)]^
        uhat_K = np.dot(hat(u), self.K)    # u^K
        K_inv = np.linalg.inv(self.K)
        A = np.dot(K_inv, (K_uhat + uhat_K))    # K_inv * ([K(u-u*)]^ + u^K)
        return A

    def ode_eq(self, s, y, method=0):
        dydt = np.empty([15, 1])
        r = np.array(y[:3])
        R = np.array([[y[3], y[4], y[5]],
                      [y[6], y[7], y[8]],
                      [y[9], y[10], y[11]]])
        u = np.array(y[12:])

        # Derivatives
        e3 = np.array([0,0,1]).T
        dr = np.dot(R, e3)
        dR = np.dot(R, hat(u))
        if method == 0:
            du = self.f_u(u)
        elif method == 1:
            du = np.dot(self.A_exact(u), u) # A = df/du -> f = A*u + C, f= u'
        else:
            du = np.dot(self.A_approx(u), u)
        #du = 0*u

        dydt[:3,:] = dr.reshape(3,1)
        dydt[3:12,:] = dR.reshape(9,1)
        dydt[12:,:] = du.reshape(3,1)

        assert dydt.shape == (15,1)
        return dydt.ravel()

    def solve(self, method):
        y_0 = np.vstack((self.r_0, self.R_0, self.u_0)).ravel()
        s = solve_ivp(lambda s, y: self.ode_eq(s, y, method), (0,self.length), y_0, method='RK23', max_step=self.step)
        ans = s.y.transpose()
        self.r = np.vstack((self.r, ans[:, (0, 1, 2)]))

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
    obs1 = Observer()
    obs2 = Observer()
    obs3 = Observer()
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    obs1.solve(0)
    obs1.plot(ax, "u\'", '-r')
    obs2.solve(1)
    obs2.plot(ax, "Au", '-b')
    obs3.solve(2)
    obs3.plot(ax, "Au approx", '-g')
    print(np.mean(np.abs(obs1.r - obs2.r), axis=0)/obs1.length)

    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()




