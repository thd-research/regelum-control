"""Contains a generic interface for systems (environments) as well as concrete systems as realizations of the former.

Remarks: 

- All vectors are treated as of type [n,]
- All buffers are treated as of type [L, n] where each row is a vector
- Buffers are updated from bottom to top

"""
from __future__ import annotations
import numpy as np


import regelum
from abc import ABC, abstractmethod
from .__utilities import rc
from typing import Optional, Union
from typing_extensions import Self


class ComposedSystem(regelum.RegelumBase):
    """Base class for composed systems.

    An instance of this class is being created automatically when applying a `@` operation on two systems.
    """

    def __init__(
        self,
        sys_left: Union[System, Self],
        sys_right: Union[System, Self],
        io_mapping: Optional[list] = None,
        output_mode="right",
    ):
        """Initialize a composed system by specifying systems to compose.

        :param sys_left: System outputs of which are to connected to the inputs of the right system
        :type sys_left: Union[System, Self]
        :param sys_right: Second system that can be connected to the inputs of the left system
        :type sys_right: Union[System, Self]
        :param io_mapping: Mapping of inputs of the right system to inputs of the left system, defaults to None
        :type io_mapping: Optional[list], optional
        :param output_mode: How to combine the result outputs, defaults to "right"
        :type output_mode: str, optional
        """
        if io_mapping is None:
            io_mapping = np.arange(min(sys_left.dim_state, sys_right.dim_inputs))

        assert output_mode in [
            "state",
            "right",
            "both",
        ], "output_mode must be 'state', 'right' or 'both'"

        if "diff_eqn" in [sys_left.system_type, sys_right.system_type]:
            self.system_type = "diff_eqn"
        else:
            self.system_type = sys_right.system_type

        self.sys_left = sys_left
        self.sys_right = sys_right
        self.parameters = sys_left.parameters | sys_right.parameters
        self.dim_state = self.sys_right.dim_state + self.sys_left.dim_state
        if output_mode == "state":
            self.dim_observation = self.sys_left.dim_state + self.sys_right.dim_state
        elif output_mode == "right":
            self.dim_observation = self.sys_right.dim_observation
        elif output_mode == "both":
            self.dim_observation = (
                self.sys_left.dim_observation + self.sys_right.dim_observation
            )

        self.rout_idx, self.occupied_idx = self.__get_routing(io_mapping)
        self.dim_inputs = (
            self.sys_right.dim_inputs
            + self.sys_left.dim_inputs
            - len(self.occupied_idx)
        )
        self.name = self.sys_left.name + " + " + self.sys_right.name

        self.free_right_input_indices = np.setdiff1d(
            np.arange(self.sys_right.dim_inputs).astype(int),
            self.occupied_idx.astype(int),
        )
        self.output_mode = output_mode
        self.forward_permutation = np.arange(self.dim_observation).astype(int)
        self.inverse_permutation = np.arange(self.dim_observation).astype(int)

    @staticmethod
    def __get_routing(io_mapping):
        io_mapping_extended = []

        for i, outputs in enumerate(io_mapping):
            assert type(outputs) in [
                tuple,
                int,
                np.int32,
                np.int64,
                None,
            ], (
                "io_mapping must be a list of ints or tuples or Nones. "
                + f"However a value of type {type(outputs)} was provided."
            )
            if not isinstance(outputs, tuple):
                outputs = (outputs,)
            elif outputs is None:
                continue
            for output in outputs:
                io_mapping_extended.append([i, output])

        io_mapping_extended = sorted(io_mapping_extended, key=lambda x: x[1])
        rout_idx, occupied_idx = rc.array(io_mapping_extended).astype(int).T
        return rout_idx, occupied_idx

    def compute_state_dynamics(
        self, time, state, inputs, _native_dim=False
    ) -> np.ndarray:
        if not _native_dim:
            return rc.force_row(
                self._compute_state_dynamics(
                    time,
                    rc.force_column(state),
                    rc.force_column(inputs),
                    _native_dim=_native_dim,
                )
            )
        else:
            return self._compute_state_dynamics(
                time, state, inputs, _native_dim=_native_dim
            )

    def get_observation(self, time, state, inputs, _native_dim=False):
        if not _native_dim:
            return rc.force_row(
                self._get_observation(
                    time,
                    rc.force_column(state),
                    rc.force_column(inputs),
                    _native_dim=_native_dim,
                )
            )
        else:
            return self._get_observation(time, state, inputs, _native_dim=_native_dim)

    def _compute_state_dynamics(self, time, state, inputs, _native_dim):
        state = rc.array(state, prototype=state)
        inputs = rc.array(inputs, prototype=state)
        state = state[self.forward_permutation]

        state_for_left, state_for_right = (
            state[: self.sys_left.dim_state],
            state[self.sys_left.dim_state :],
        )
        inputs_for_left = inputs[: self.sys_left.dim_inputs]
        dstate_of_left = rc.squeeze(
            self.sys_left.compute_state_dynamics(
                time=time,
                state=state_for_left,
                inputs=inputs_for_left,
                _native_dim=_native_dim,
            )
        )
        outputs_of_left = rc.squeeze(
            self.sys_left.get_observation(
                time=time,
                state=state_for_left,
                inputs=inputs_for_left,
                _native_dim=_native_dim,
            )
        )

        inputs_for_right = rc.zeros(
            self.sys_right.dim_inputs,
            prototype=(state, inputs),
        )
        inputs_for_right[self.occupied_idx] = outputs_of_left[self.rout_idx]
        inputs_for_right[self.free_right_input_indices] = rc.reshape(
            inputs[self.sys_left.dim_inputs :],
            rc.shape(inputs_for_right[self.free_right_input_indices]),
        )

        dstate_of_right = rc.squeeze(
            self.sys_right.compute_state_dynamics(
                time=time,
                state=state_for_right,
                inputs=inputs_for_right,
                _native_dim=_native_dim,
            )
        )
        final_dstate_vector = rc.hstack((dstate_of_left.T, dstate_of_right.T)).T

        assert (
            final_dstate_vector is not None
        ), f"final dstate_vector of system {self.name} is None"
        final_dstate_vector = final_dstate_vector[self.inverse_permutation]
        if not _native_dim:
            final_dstate_vector = rc.force_row(final_dstate_vector)
        return final_dstate_vector

    def _get_observation(self, time, state, inputs, _native_dim):
        state = rc.array(state, prototype=state)
        inputs = rc.array(inputs, prototype=state)

        inputs_for_left = inputs[: self.sys_left.dim_inputs]
        state_for_left, state_for_right = (
            state[: self.sys_left.dim_state],
            state[self.sys_left.dim_state :],
        )
        outputs_of_left = rc.squeeze(
            self.sys_left.get_observation(
                time=time,
                state=state_for_left,
                inputs=inputs_for_left,
                _native_dim=_native_dim,
            )
        )

        inputs_for_right = rc.zeros(
            self.sys_right.dim_inputs,
            prototype=(state, inputs),
        )
        inputs_for_right[self.occupied_idx] = outputs_of_left[self.rout_idx]
        inputs_for_right[self.free_right_input_indices] = rc.reshape(
            inputs[self.sys_left.dim_inputs :],
            rc.shape(inputs_for_right[self.free_right_input_indices]),
        )
        outputs_of_right = self.sys_right.get_observation(
            time=time,
            state=state_for_right,
            inputs=inputs_for_right,
            _native_dim=_native_dim,
        )
        if self.output_mode == "right":
            output = outputs_of_right
        elif self.output_mode == "state":
            output = state
        elif self.output_mode == "both":
            output = rc.concatenate((state_for_left, state_for_right))
        else:
            raise NotImplementedError

        ## TODO: implement 'preserve' mode

        return output

    def receive_action(self, action):
        self.inputs = action

    def update_system_parameters(self, inputs):
        assert isinstance(inputs, dict)
        self.sys_left.update_system_parameters(inputs)
        self.sys_right.update_system_parameters(inputs)

    # TODO: get rid of it
    def compute_closed_loop_rhs(self, time, state):
        action = self.inputs

        rhs_full_state = self.compute_state_dynamics(time, state, action)

        return rhs_full_state

    def receive_state(self, state):
        self.state = state

    def reset(self):
        pass

    def permute_state(self, permutation: Union[list, np.array]) -> Self:
        """Permute an order at which the system outputs are returned.

        :param permutation: Permutation represented as an array of indices
        :type permutation: Union[list, np.array]
        :return: link to self
        :rtype: Self
        """
        self.forward_permutation = rc.array(permutation).astype(int)
        self.inverse_permutation = self.get_inverse_permutation(permutation)
        return self

    def get_inverse_permutation(self, permutation):
        self.current_permutation = permutation
        permutation = np.asanyarray(permutation)
        inverse_permutation = np.empty_like(permutation)
        inverse_permutation[permutation] = np.arange(permutation.size)
        return inverse_permutation

    def compose(self, sys_right, io_mapping=None, output_mode="state"):
        return ComposedSystem(
            self, sys_right, io_mapping=io_mapping, output_mode=output_mode
        )

    def __matmul__(self, sys_right):
        return self.compose(sys_right)


class System(regelum.RegelumBase, ABC):
    """Base class for controlled systems implementation."""

    _name: Optional[str] = None
    _system_type: Optional[str] = None
    _dim_state: Optional[int] = None
    _dim_inputs: Optional[int] = None
    _dim_observation: Optional[int] = None
    _parameters = {}

    def __init__(
        self,
        system_parameters_init=None,
        state_init: Optional[np.ndarray] = None,
        inputs_init: Optional[np.ndarray] = None,
    ):
        """Initialize an instance of a system.

        :param system_parameters_init: Set system parameters manually, defaults to {}
        :type system_parameters_init: dict, optional
        :param state_init: Set initial state manually, defaults to None
        :type state_init: Optional[np.ndarray], optional
        :param inputs_init: Set initial inputs manually, defaults to None
        :type inputs_init: Optional[np.ndarray], optional
        """
        if system_parameters_init is None:
            system_parameters_init = {}
        assert self.name is not None
        assert self.system_type is not None
        assert self.dim_state is not None
        assert self.dim_inputs is not None
        assert self.dim_observation is not None
        assert isinstance(
            system_parameters_init, dict
        ), "system_parameters_init should be a dict"

        if system_parameters_init:
            self._parameters.update(system_parameters_init)

        self.system_parameters_init = self._parameters

        if state_init is None:
            self.state = rc.zeros(self.dim_state)
        else:
            self.state = state_init

        if inputs_init is None:
            self.inputs = rc.zeros(self.dim_inputs)
        else:
            self.inputs = inputs_init

    @property
    def name(self) -> str:
        if self._name is None:
            raise ValueError("system._name should be set")
        return self._name

    @property
    def system_type(self) -> str:
        if self._system_type is None:
            raise ValueError("system._system_type should be set")
        return self._system_type

    @property
    def dim_state(self) -> int:
        if self._dim_state is None:
            raise ValueError("system._dim_state should be set")
        return self._dim_state

    @property
    def dim_observation(self) -> int:
        if self._dim_observation is None:
            raise ValueError("system._dim_observation should be set")
        return self._dim_observation

    @property
    def dim_inputs(self):
        if self._dim_inputs is None:
            raise ValueError("system._dim_inputs should be set")
        return self._dim_inputs

    @property
    def parameters(self):
        return self._parameters

    def compute_state_dynamics(
        self, time, state, inputs, _native_dim=False
    ) -> np.ndarray:
        if not _native_dim:
            return rc.force_row(
                self._compute_state_dynamics(
                    time, rc.force_column(state), rc.force_column(inputs)
                )
            )
        else:
            return self._compute_state_dynamics(time, state, inputs)

    @abstractmethod
    def _compute_state_dynamics(time, state, inputs):
        pass

    def get_observation(self, time, state, inputs, _native_dim=False):
        if not _native_dim:
            return rc.force_row(
                self._get_observation(
                    time, rc.force_column(state), rc.force_column(inputs)
                )
            )
        else:
            return self._get_observation(time, state, inputs)

    def _get_observation(self, time, state, inputs):
        return state

    def receive_action(self, action):
        self.inputs = action

    def receive_state(self, state):
        self.state = state

    def update_system_parameters(self, inputs):
        assert isinstance(inputs, dict)
        self._parameters.update(inputs)
        return self.parameters

    def compute_closed_loop_rhs(self, time, state):
        """Legacy code.

        Right-hand side of the closed-loop system description.
        Combines everything into a single vector that corresponds to the right-hand side of the closed-loop system description for further use by simulators.

        Attributes
        ----------
        state_full : : vector
            Current closed-loop system state

        """
        action = self.inputs

        rhs_full_state = self.compute_state_dynamics(time, state, action)

        return rhs_full_state

    def compose(self, sys_right, io_mapping=None, output_mode="state"):
        return ComposedSystem(
            self, sys_right, io_mapping=io_mapping, output_mode=output_mode
        )

    def __matmul__(self, sys_right):
        return self.compose(sys_right)

    def reset(self):
        self.update_system_parameters(self.system_parameters_init)


class KinematicPoint(System):
    """System representing Kinematic Point (omnibot)."""

    _name = "kinematic-point"
    _system_type = "diff_eqn"
    _dim_state = 2
    _dim_inputs = 2
    _dim_observation = 2

    def _compute_state_dynamics(self, time, state, inputs):
        Dstate = rc.zeros(
            self.dim_state,
            prototype=(state, inputs),
        )

        for i in range(rc.shape(inputs)[0]):
            Dstate[i] = inputs[i]

        return Dstate


class InvertedPendulumPID(System):
    """System class: mathematical pendulum."""

    _name = "inverted-pendulum"
    _system_type = "diff_eqn"
    _dim_state = 2
    _dim_inputs = 1
    _dim_observation = 3
    _parameters = {"m": 1, "g": 9.8, "l": 1}

    def __init__(self, *args, **kwargs):
        """Initialize an instance of an Inverted Pendulum, which gives an observation suitable for PID controller."""
        super().__init__(*args, **kwargs)

        self.time_old = 0
        self.integral_alpha = 0

    def _compute_state_dynamics(self, time, state, inputs):
        Dstate = rc.zeros(
            self.dim_state,
            prototype=(state, inputs),
        )

        m, g, l = (
            self.parameters["m"],
            self.parameters["g"],
            self.parameters["l"],
        )

        Dstate[0] = state[1]
        Dstate[1] = g / l * rc.sin(state[0]) + inputs[0] / (m * l**2)

        return Dstate

    def _get_observation(self, time, state, inputs):
        delta_time = time - self.time_old if time is not None else 0
        self.integral_alpha += delta_time * state[0]

        return rc.hstack([state[0], self.integral_alpha, state[1]])

    def reset(self):
        self.time_old = 0
        self.integral_alpha = 0


class InvertedPendulumPD(InvertedPendulumPID):
    """System class: ordinary mathematical pendulum."""

    _dim_observation = 2

    def _get_observation(self, time, state, inputs):
        return rc.hstack([state[0], state[1]])


class ThreeWheeledRobot(System):
    r"""System class: 3-wheeled robot with dynamical actuators.

    Description
    -----------
    Three-wheel robot with dynamical pushing force and steering torque (a.k.a. ENDI - extended non-holonomic double integrator) [[1]_]

    .. math::
        \begin{array}{ll}
                        \dot x_с & = v \cos \angle \newline
                        \dot y_с & = v \sin \angle \newline
                        \dot \angle & = \omega \newline
                        \dot v & = \left( \frac 1 m F + q_1 \right) \newline
                        \dot \omega & = \left( \frac 1 I M + q_2 \right)
        \end{array}

    **Variables**

    | :math:`x_с` : state-coordinate [m]
    | :math:`y_с` : observation-coordinate [m]
    | :math:`\angle` : turning angle [rad]
    | :math:`v` : speed [m/s]
    | :math:`\omega` : revolution speed [rad/s]
    | :math:`F` : pushing force [N]
    | :math:`M` : steering torque [Nm]
    | :math:`m` : robot mass [kg]
    | :math:`I` : robot moment of inertia around vertical axis [kg m\ :sup:`2`]
    | :math:`disturb` : actuator disturbance (see :func:`~RLframe.system.disturbDyn`). Is zero if ``is_disturb = 0``

    :math:`state = [x_c, y_c, \angle, v, \omega]`

    :math:`inputs = [F, M]`

    ``pars`` = :math:`[m, I]`

    References
    ----------
    .. [1] W. Abbasi, F. urRehman, and I. Shah. “Backstepping based nonlinear adaptive control for the extended
        nonholonomic double integrator”. In: Kybernetika 53.4 (2017), pp. 578–594

    """

    _name = "three-wheeled-robot"
    _system_type = "diff_eqn"
    _dim_state = 5
    _dim_inputs = 2
    _dim_observation = 5
    _parameters = {"m": 10, "I": 1}

    def _compute_state_dynamics(self, time, state, inputs):
        Dstate = rc.zeros(
            self.dim_state,
            prototype=(state, inputs),
        )

        m, I = self.parameters["m"], self.parameters["I"]

        Dstate[0] = state[3] * rc.cos(state[2])
        Dstate[1] = state[3] * rc.sin(state[2])
        Dstate[2] = state[4]
        Dstate[3] = 1 / m * inputs[0]
        Dstate[4] = 1 / I * inputs[1]

        return Dstate


class Integrator(System):
    """System yielding Non-holonomic double integrator when composed with kinematic thre-wheeled robot."""

    _name = "integral-parts"
    _system_type = "diff_eqn"
    _dim_state = 2
    _dim_inputs = 2
    _dim_observation = 2
    _parameters = {"m": 10, "I": 1}

    def _compute_state_dynamics(self, time, state, inputs):
        Dstate = rc.zeros(
            self.dim_state,
            prototype=(state, inputs),
        )

        m, I = self.parameters["m"], self.parameters["I"]

        Dstate[0] = 1 / m * inputs[0]
        Dstate[1] = 1 / I * inputs[1]

        return Dstate


class ThreeWheeledRobotNI(System):
    """System class: 3-wheel robot with static actuators (the NI - non-holonomic integrator)."""

    _name = "three-wheeled-robot-ni"
    _system_type = "diff_eqn"
    _dim_state = 3
    _dim_inputs = 2
    _dim_observation = 3

    def _compute_state_dynamics(self, time, state, inputs):
        Dstate = rc.zeros(self.dim_state, prototype=(state, inputs))

        Dstate[0] = inputs[0] * rc.cos(state[2])
        Dstate[1] = inputs[0] * rc.sin(state[2])
        Dstate[2] = inputs[1]

        return Dstate


class TwoTank(System):
    """Two-tank system with nonlinearity."""

    _name = "two-tank"
    _system_type = "diff_eqn"
    _dim_state = 2
    _dim_inputs = 1
    _dim_observation = 2
    _parameters = {"tau1": 18.4, "tau2": 24.4, "K1": 1.3, "K2": 1.0, "K3": 0.2}

    def _compute_state_dynamics(self, time, state, inputs):
        tau1, tau2, K1, K2, K3 = (
            self.parameters["tau1"],
            self.parameters["tau2"],
            self.parameters["K1"],
            self.parameters["K2"],
            self.parameters["K3"],
        )

        Dstate = rc.zeros(
            self.dim_state,
            prototype=(state, inputs),
        )
        Dstate[0] = 1 / (tau1) * (-state[0] + K1 * inputs[0])
        Dstate[1] = 1 / (tau2) * (-state[1] + K2 * state[0] + K3 * state[1] ** 2)

        return Dstate


class GridWorld(System):
    """A simple 2-dimensional grid world with five actions: left, right, up, down and do nothing.

    The inputs encoding rule is as follows: right, left, up, down, do nothing -> 0, 1, 2, 3, 4.
    """

    def __init__(self, dims, terminal_state):
        """Initialize an instance of GridWorld.

        :param dims: grid dimensions (height, width)
        :type dims: tuple
        :param terminal_state: coordinates of goal cell
        :type terminal_state: list
        """
        self.dims = dims
        self.terminal_state = terminal_state

    def _compute_state_dynamics(self, current_state, inputs):
        if tuple(self.terminal_state) == tuple(current_state):
            return current_state
        if inputs == 0:
            if current_state[1] < self.dims[1] - 1:
                return (current_state[0], current_state[1] + 1)
        elif inputs == 2:
            if current_state[0] > 0:
                return (current_state[0] - 1, current_state[1])
        elif inputs == 1:
            if current_state[1] > 0:
                return (current_state[0], current_state[1] - 1)
        elif inputs == 3:
            if current_state[0] < self.dims[0] - 1:
                return (current_state[0] + 1, current_state[1])
        return current_state


class CartPole(System):
    """Cart pole system without friction."""

    _name = "cartpole"
    _system_type = "diff_eqn"
    _dim_state = 4
    _dim_inputs = 1
    _dim_observation = 4
    _parameters = {"m_c": 0.1, "m_p": 2.0, "g": 9.81, "l": 0.5}

    def _compute_state_dynamics(self, time, state, inputs, disturb=None):
        Dstate = rc.zeros(
            self.dim_state,
            prototype=(state, inputs),
        )

        m_c, m_p, g, l = (
            self.parameters["m_c"],
            self.parameters["m_p"],
            self.parameters["g"],
            self.parameters["l"],
        )
        theta = state[0]
        theta_dot = state[2]
        x_dot = state[3]

        sin_theta = rc.sin(theta)
        cos_theta = rc.cos(theta)

        # Dstate[0] = theta_dot

        # Dstate[1] = x_dot

        # Dstate[2] = (
        #     (
        #         g * rc.sin(theta)
        #         - rc.cos(theta)
        #         * (inputs[0] + m_p * l * theta_dot**2 * rc.sin(theta))
        #         / (m_c + m_p)
        #     )
        #     / l
        #     / (4 / 3 - m_p * (rc.cos(theta) ** 2) / (m_c + m_p))
        # )
        # Dstate[3] = (
        #     inputs[0]
        #     + m_p
        #     * l
        #     * (
        #         theta_dot**2 * rc.sin(theta)
        #         - Dstate[0] * rc.cos(theta)
        #     )
        # ) / (m_c + m_p)

        Dstate[0] = theta_dot

        Dstate[1] = x_dot

        Dstate[3] = (
            -m_p * g * cos_theta * sin_theta
            - m_p * l * theta_dot**2 * sin_theta
            + inputs[0]
        ) / (m_c + m_p * sin_theta**2)

        Dstate[2] = -g / l * sin_theta + Dstate[3] / l * cos_theta

        return Dstate

    def _get_observation(self, time, state, inputs):
        theta = state[0]
        x = state[1]
        theta_dot = state[2]
        x_dot = state[3]

        theta_observed = theta - rc.floor(theta / (2 * np.pi)) * 2 * np.pi
        if theta_observed > np.pi:
            theta_observed = theta_observed - 2 * np.pi

        return rc.array([theta_observed, x, theta_dot, x_dot])


class LunarLander(System):
    """Lunar lander system.

    link: https://web.aeromech.usyd.edu.au/AMME3500/Course_documents/material/tutorials/Assignment%204%20Lunar%20Lander%20Solution.pdf.
    """

    _name = "lander"
    _system_type = "diff_eqn"
    _dim_state = 6
    _dim_inputs = 2
    _dim_observation = 6
    _parameters = {"m": 10, "J": 3.0, "g": 1.625, "a": 1, "r": 0.5, "sigma": 0.1}

    def __init__(self, *args, **kwargs):
        """Initialize an instance of LunarLander by specifying relevant physical parameters."""
        super().__init__(*args, **kwargs)

        self.name = "lander"
        self.a = 1
        self.r = 1
        self.alpha = np.arctan(self.a / self.r)
        self.l = np.sqrt(self.a**2 + self.r**2)
        self.sigma = 1
        self.state_cache = []
        self.is_landed = False

    def _compute_state_dynamics(self, time, state, inputs, disturb=None):
        Dstate_before_landing = rc.zeros(
            self.dim_state,
            prototype=(state, inputs),
        )

        m, J, g = (
            self.parameters["m"],
            self.parameters["J"],
            self.parameters["g"],
        )

        theta = state[2]
        x_dot = state[3]
        y_dot = state[4]
        theta_dot = state[5]

        left_support, right_support = self.compute_supports_geometry(state[:2], theta)

        self.is_landed = (
            rc.if_else(left_support[1] <= 0, 1, 0)
            + rc.if_else(right_support[1] <= 0, 1, 0)
        ) > 0

        F_l = inputs[0] * (1 - self.is_landed)
        F_t = inputs[1] * (1 - self.is_landed)

        self.is_landed_left = rc.if_else(left_support[1] <= 0, 1, 0)
        self.is_landed_right = rc.if_else(right_support[1] <= 0, 1, 0)
        self.is_landed_vertex = rc.if_else(state[1] <= 0, 1, 0)
        self.is_freezed = (
            self.is_landed_left * self.is_landed_right + self.is_landed_vertex
        ) > 0
        self.is_landed = (
            self.is_landed_left + self.is_landed_right + self.is_landed_vertex
        ) > 0

        Dstate_before_landing[0] = x_dot
        Dstate_before_landing[1] = y_dot
        Dstate_before_landing[2] = theta_dot
        Dstate_before_landing[3] = 1 / m * (F_l * rc.cos(theta) - F_t * rc.sin(theta))
        Dstate_before_landing[4] = (
            1 / m * (F_l * rc.sin(theta) + F_t * rc.cos(theta)) - g
        )
        Dstate_before_landing[5] = (4 * F_l) / J

        Dstate_landed_right = self._compute_pendulum_dynamics(
            # x=x,
            # y=y,
            angle=-theta - self.alpha,
            angle_dot=theta_dot,
            prototype=(state, inputs),
        )

        Dstate_landed_left = self._compute_pendulum_dynamics(
            # x=x,
            # y=y,
            angle=self.alpha - theta,
            angle_dot=theta_dot,
            prototype=(state, inputs),
        )

        # Check if any of the two lander's supports touched the ground. If yes, freeze the state.

        Dstate = (1 - self.is_freezed) * (
            (1 - self.is_landed) * Dstate_before_landing
            + self.is_landed
            * (
                self.is_landed_right * Dstate_landed_right
                + self.is_landed_left * Dstate_landed_left
            )
        )

        return Dstate

    def _compute_pendulum_dynamics(self, angle, angle_dot, prototype):
        Dstate = rc.zeros(
            self.dim_state,
            prototype=prototype,
        )
        m, J, g = self.pars

        x = self.l * rc.sin(angle)
        y = self.l * rc.cos(angle)

        Dstate[5] = g / self.l**2 * x

        # Dstate[0] = angle_dot * y
        # Dstate[1] = -angle_dot * x
        # Dstate[2] = angle_dot
        # Dstate[3] = y * Dstate[5] - angle_dot**2 * x
        # Dstate[4] = -x * Dstate[5] - angle_dot**2 * y
        # angle_dot = 1
        # Dstate[5] = 0.0
        Dstate[0] = angle_dot * y
        Dstate[1] = -angle_dot * x
        Dstate[2] = -angle_dot
        Dstate[3] = y * Dstate[5] - angle_dot**2 * x
        Dstate[4] = -x * Dstate[5] - angle_dot**2 * y

        return Dstate

    def compute_supports_geometry(self, xi, theta):
        A = rc.zeros((2, 2), prototype=xi)
        xi_2 = rc.zeros(2, prototype=xi)
        xi_3 = rc.zeros(2, prototype=xi)

        A[0, 0] = rc.cos(theta)
        A[0, 1] = -rc.sin(theta)
        A[1, 0] = rc.sin(theta)
        A[1, 1] = rc.cos(theta)

        a, r = self.parameters["a"], self.parameters["r"]
        xi_2[0] = xi[0] - a
        xi_2[1] = xi[1] - r
        xi_3[0] = xi[0] + a
        xi_3[1] = xi[1] - r

        xi_2_d = xi_2 - xi
        xi_3_d = xi_3 - xi

        xi_2_d_rot = A @ xi_2_d
        xi_3_d_rot = A @ xi_3_d
        xi_2_new = xi + xi_2_d_rot
        xi_3_new = xi + xi_3_d_rot
        return xi_2_new, xi_3_new

    def compute_reaction(
        self,
        r,
        r_support,
    ):
        m, _, g, sigma = (
            self.parameters["m"],
            self.parameters["J"],
            self.parameters["g"],
            self.parameters["sigma"],
        )
        lvl = r_support[1]
        e = (r - r_support) / rc.sqrt(rc.norm_2(r - r_support))
        reaction = rc.if_else(
            lvl <= 0,
            e * rc.dot(e, m * g * rc.array([0, 1])) * lvl * sigma,
            rc.array([0.0, 0.0]),
        )
        return -reaction
