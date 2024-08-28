'''
Implementation of two-link manipulator that follows example A.10 fron Nonlinear Controls by Hassan K. Khalil
'''

from manipulator_adaptive_control import RBFNNAdaptiveController
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


class TwoLinkManipulator:

    def __init__(self):
        self.a1 = 200.01
        self.a2 = 23.5
        self.a3 = 122.5
        self.a4 = 25
        self.b1 = 784.8
        self.b2 = 245.25

        self.x = np.zeros(4)  #x = [q, qdot]

    def _calc_xdot(self, tau):
        pass

    def _calc_M(self, q):
        M = np.zeros((2, 2))
        M[0, 0] = self.a1 + 2 * self.a4 * np.cos(q[1])
        M[0, 1] = self.a2 + self.a4 * np.cos(q[1])
        M[1, 0] = self.a2 + self.a4 * np.cos(q[1])
        M[1, 1] = self.a3
        return M

    def _calc_C(self, q, qdot):
        C = np.zeros((2, 2))
        C[0, 0] = -qdot[1]
        C[0, 1] = -(qdot[0] + qdot[1])
        C[1, 0] = qdot[0]
        C[1, 1] = 0
        return C * self.a4 * np.sin(q[1])

    def _calc_g(self, q):
        g = np.zeros(2)
        g[0] = self.b1 * np.sin(q[0]) + self.b2 * np.cos(q[0] + q[1])
        g[1] = self.b2 * np.cos(q[0] + q[1])

        return g

    def step(self, tau, dt):
        q = self.x[:2]
        qdot = self.x[2:]

        M = self._calc_M(q)
        C = self._calc_C(q, qdot)
        g = self._calc_g(q)

        xdot = np.zeros(4)
        xdot[:2] = qdot
        xdot[2:] = np.linalg.inv(M) @ (tau - C @ qdot - g)

        self.x += xdot * dt


if __name__ == '__main__':

    controller = RBFNNAdaptiveController(
        num_gen_coords=2,
        numberOfRBFCenters=20,
        RBFmins=np.array([-100] * 8),
        RBFmaxes=np.array([100] * 8),
        zeta=np.array([1, 1]),
        time_constant=np.array([.5, .5]),
        Lambda=50.0,
        Gamma=20,
        KD=500.0,
        ctrl_dt=0.005,
    )

    manipulator = TwoLinkManipulator()

    def step_commands(t):
        if t < 5:
            return np.array([np.pi / 2, -np.pi / 2])
        elif t < 10:
            return np.array([-np.pi / 2, np.pi / 2])
        else:
            return np.zeros(2)

    q_hist = []
    qdot_hist = []
    tau_hist = []
    q_track_hist = []

    q_des = np.array([np.pi / 2, -np.pi / 2])
    time = np.arange(0, 15, 0.005)
    for t in time:
        q_des = step_commands(t)
        q = deepcopy(manipulator.x[:2])
        qdot = deepcopy(manipulator.x[2:])
        tau, _, _, _, _, q_track, qd_track, qdd_track = controller.solve_for_next_u(
            q, qdot, q_des)

        q_hist.append(q)
        q_track_hist.append(q_track)
        qdot_hist.append(qdot)
        tau_hist.append(tau)

        manipulator.step(tau, 0.005)

    q_hist = np.array(q_hist)
    qdot_hist = np.array(qdot_hist)
    tau_hist = np.array(tau_hist)
    q_track_hist = np.array(q_track_hist)

    plt.plot(time, q_hist[:, 0], label='q1')
    plt.plot(time, q_hist[:, 1], label='q2')
    plt.plot(time, q_track_hist[:, 0], '--', label='q1_track')
    plt.plot(time, q_track_hist[:, 1], '--', label='q2_track')
    plt.legend()
    plt.show()
