from regelum.policy import Policy
from regelum import CasadiOptimizerConfig
import numpy as np


class ThreeWheeledRobotNomial(Policy):
    def __init__(
        self,
        action_bounds: list[list[float]],
        kappa_params: list[float] = [2, 15, -1.50],
        eps=0.01,
    ):
        super().__init__()
        self.action_bounds = action_bounds
        # An epsilon for numerical stability
        self.eps = eps
        self.update_kappa(*kappa_params)

    def update_kappa(self, k_rho, k_alpha, k_beta):
        # Parameters for gazebo
        self.k_rho = k_rho
        self.k_alpha = k_alpha  
        self.k_beta = k_beta

    def get_action(self, observation: np.ndarray):
        x_robot = observation[0, 0]
        y_robot = observation[0, 1]
        theta = observation[0, 2]

        x_goal = 0
        y_goal = 0
        theta_goal = 0

        error_x = x_goal - x_robot
        error_y = y_goal - y_robot
        error_theta = theta_goal - theta

        rho = np.sqrt(error_x**2 + error_y**2)
        alpha = -theta + np.arctan2(error_y, error_x)
        beta = error_theta - alpha

        w = self.k_alpha*alpha + self.k_beta*beta
        v = self.k_rho*rho

        while alpha > np.pi:
            alpha -= 2* np.pi

        while alpha < -np.pi:
            alpha += 2* np.pi

        if -np.pi < alpha <= -np.pi / 2 or np.pi / 2 < alpha <= np.pi:
            v = -v
        
        return np.array([[v, w]])