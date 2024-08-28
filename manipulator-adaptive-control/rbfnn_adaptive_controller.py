import numpy as np
import numpy.typing as npt
from typing import Callable, Tuple

from scipy.stats import qmc
import scipy

from copy import deepcopy


def initialize_lhs(lb, ub, num_points):
    # generate LHS points
    sample = qmc.LatinHypercube(d=lb.shape[0], seed=7).random(num_points)
    scaled = qmc.scale(sample, lb, ub)
    return scaled


def calculate_max_distance(points):
    max_distance = 0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distance = np.linalg.norm(points[i] - points[j])
            if distance > max_distance:
                max_distance = distance
    return max_distance


class RBFNNAdaptiveController:
    def __init__(
        self,
        num_gen_coords: int,
        numberOfRBFCenters: int,
        RBFmins: npt.NDArray,
        RBFmaxes: npt.NDArray,
        zeta=npt.NDArray,
        time_constant=npt.NDArray,
        Lambda=2.0,
        Gamma=0.1,
        KD=1.0,
        ctrl_dt=0.005,
    ):
        self.n = num_gen_coords
        self.N = self.n * 4
        self.M = numberOfRBFCenters

        assert len(RBFmins) == self.N
        assert len(RBFmaxes) == self.N
        assert len(zeta) == self.n
        assert len(time_constant) == self.n

        self.RBFmins = RBFmins  # should be same length as x in Phi(x)
        self.RBFmaxes = RBFmaxes  # should be same length as x in Phi(x)

        self.theta_hat = 0 * np.random.uniform(
            -1, 1, size=(self.M + 1, self.n))
        self.Gamma = np.eye(self.M + 1) * np.atleast_1d(Gamma)
        self.Lambda = np.eye(self.n) * np.atleast_1d(Lambda)

        self.Lambda_2 = np.eye(self.M) * 1
        self.Kd = np.eye(self.n) * KD
        self.dt = ctrl_dt
        self.zeta = zeta
        self.time_constant = time_constant

        self._create_desired_system(zeta, time_constant, ctrl_dt)

        self.center_hat = initialize_lhs(RBFmins, RBFmaxes,
                                         self.M)  #self.M x self.N

        # dMax is the max distance between centers
        self.dMax = calculate_max_distance(self.center_hat)
        self.width = self.M / self.dMax**2

        self.q_des = np.zeros([self.n])
        self.qdot_des = np.zeros([self.n])
        self.qddot_des = np.zeros([self.n])

    def _create_desired_system(self, zeta, tau, dt):
        # A_ref (2*self.n x 2*self.n)
        # B_ref (2*self.n x num_gen_forces)

        # make self.n parameters, from second order system dynamics
        m = 1.0
        b = 2 * m / tau
        k = (b / (2 * zeta))**2

        self.x_des = np.zeros([2 * self.n])
        self.xdot_des = np.zeros([2 * self.n])

        self.Ades_stack = np.zeros([2 * self.n, 2 * self.n])
        self.Bdes_stack = np.zeros([2 * self.n, self.n])
        self.Ad_des_stack = np.zeros([2 * self.n, 2 * self.n])
        self.Bd_des_stack = np.zeros([2 * self.n, self.n])

        # create Aref, Bref, Ad_ref, Bd_ref for each generalized coordiate
        for i in range(self.n):
            Ades = np.array([
                [-b[i] / m, -k[i] / m],
                [1, 0],
            ])
            Bdes = np.array([[k[i] / m], [0]])

            # discrete time, see https://en.wikipedia.org/wiki/Discretization#Derivation
            Ad_des = scipy.linalg.expm(Ades * dt)
            Bd_des = np.matmul(
                np.linalg.inv(Ades),
                np.matmul(Ad_des - np.eye(Ad_des.shape[0]), Bdes))

            self.Ades_stack[i * 2:i * 2 + 2, i * 2:i * 2 + 2] = Ades
            self.Bdes_stack[i * 2:i * 2 + 2, i] = Bdes.flatten()
            self.Ad_des_stack[i * 2:i * 2 + 2, i * 2:i * 2 + 2] = Ad_des
            self.Bd_des_stack[i * 2:i * 2 + 2, i] = Bd_des.flatten()

    def _update_desired_trajectory(self, u: npt.NDArray):
        # x_des (2*self.n x 1)
        # r (self.n x 1)
        # xdot_des = [qdd, qd] * self.n
        # x_des = [qd, q] * self.n
        self.xdot_des = self.Ades_stack @ self.x_des + self.Bdes_stack @ u
        self.x_des = self.Ad_des_stack @ self.x_des + self.Bd_des_stack @ u

        qddot_des = self.xdot_des[::2]
        qdot_des = self.xdot_des[1::2]
        q_des = self.x_des[1::2]

        return q_des, qdot_des, qddot_des

    def _calc_regressor(
        self,
        q: npt.NDArray,
        qdot: npt.NDArray,
        qdot_ref: npt.NDArray,
        qddot_ref: npt.NDArray,
    ):
        assert len(q) == self.n
        assert len(qdot) == self.n
        assert len(qdot_ref) == self.n
        assert len(qddot_ref) == self.n

        x = np.hstack([q, qdot, qdot_ref, qddot_ref])
        norms = np.linalg.norm(x - self.center_hat, axis=1)
        Phi = np.exp(-self.width * norms**2)

        return np.append(
            Phi, 1.0
        )  # this 1 is critical for performance. Its a bias term that lets torques be non-zero centered.
        # return Phi

    # having bias is not necessary, the weights CAN compensate for it, but this is practically much harder since it rquires a higher adaptation rate.

    def _calc_refs(
        self,
        q: npt.NDArray[np.float64],
        qdot: npt.NDArray[np.float64],
        q_des: npt.NDArray[np.float64],
        qdot_des: npt.NDArray[np.float64],
        qddot_des: npt.NDArray[np.float64],
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """
        Equations 7b,7c in Slotine paper. Note that q_r is not
        explicitly needed in control law, so it is not calculated
        here.


        Args:
            q (npt.NDArray[np.float64]): joint angles
            qdot (npt.NDArray[np.float64]): joint vel
            q_d (npt.NDArray[np.float64]): desired joint angles
            qdot_d (npt.NDArray[np.float64]): desired joint vel
            qddot_d (npt.NDArray[np.float64]): desired joint accel

        Returns:
            tuple: (reference velocity, reference acceleration)
        """

        qtilde_dot = qdot - qdot_des
        qddot_ref = qddot_des - self.Lambda.dot(qtilde_dot)
        qtilde = q - q_des
        qdot_ref = qdot_des - self.Lambda.dot(qtilde)
        return qdot_ref, qddot_ref

    def _update_weights(self, s: npt.NDArray, Phi: npt.NDArray) -> None:
        thetaDot = -self.Gamma @ np.outer(Phi, s)
        self.theta_hat = self.theta_hat + thetaDot * self.dt

    def solve_for_next_u(
        self,
        q: npt.NDArray[np.float64],
        qdot: npt.NDArray[np.float64],
        q_des: npt.NDArray[np.float64],
        qdot_des: npt.NDArray[np.float64] = None,
        qddot_des: npt.NDArray[np.float64] = None,
        adapt=True,
    ) -> npt.NDArray[np.float64]:

        # given qd_des and qdd_des from somewhere, I calculate a desired trajectory. A conventient
        # choice for obtaining qd_des and qdd_des is to use a critically damped 2nd roder system.
        # if qdot_des and qddot_des are not provided, I will use the internal trajectory generator to calculate them.
        if qdot_des is None or qddot_des is None:
            qdot_ref, qddot_ref = self._calc_refs(q, qdot, self.q_des,
                                                  self.qdot_des,
                                                  self.qddot_des)

            self.q_des, self.qdot_des, self.qddot_des = self._update_desired_trajectory(
                q_des)

            #copy for returning
            q_des = deepcopy(self.q_des)
            qdot_des = deepcopy(self.qdot_des)
            qddot_des = deepcopy(self.qddot_des)
        else:
            qdot_ref, qddot_ref = self._calc_refs(q, qdot, q_des, qdot_des,
                                                  qddot_des)

        Phi = self._calc_regressor(q, qdot, qdot_ref, qddot_ref)
        s = qdot - qdot_ref

        if adapt:
            error = q_des - q
            # if np.any(np.abs(np.degrees(error)) > 0.005):
            # print(f"adapting, errpr:{error}")
            self._update_weights(s, Phi)

        tauFF = self.theta_hat.T @ Phi
        tauPD = self.Kd @ s
        tau = tauFF - tauPD

        return tau, s, self.theta_hat, tauFF, tauPD, q_des, qdot_des, qddot_des

    def sat(self, s):
        y = 1 * s
        return np.clip(y, -1, 1)
