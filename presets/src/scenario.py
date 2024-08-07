import numpy as np
import torch

from regelum.data_buffers import DataBuffer

from regelum.policy import PolicyReinforce
from typing import Optional, Dict, Any, Callable, Type
from regelum.objective import RunningObjective
from regelum.data_buffers import DataBuffer
from regelum.simulator import Simulator
from regelum.observer import Observer
from regelum.model import (
    PerceptronWithTruncatedNormalNoise,
    ModelPerceptron
)
from regelum.scenario import RLScenario, get_policy_gradient_kwargs, CALF, PPO


class MyPPO(PPO):
    def __init__(
        self,
        policy_model: PerceptronWithTruncatedNormalNoise,
        critic_model: ModelPerceptron,
        sampling_time: float,
        running_objective: RunningObjective,
        simulator: Simulator,
        critic_n_epochs: int,
        policy_n_epochs: int,
        critic_opt_method_kwargs: Dict[str, Any],
        policy_opt_method_kwargs: Dict[str, Any],
        critic_opt_method: Type[torch.optim.Optimizer] = torch.optim.Adam,
        policy_opt_method: Type[torch.optim.Optimizer] = torch.optim.Adam,
        running_objective_type: str = "cost",
        critic_td_n: int = 1,
        cliprange: float = 0.2,
        discount_factor: float = 0.7,
        observer: Optional[Observer] = None,
        N_episodes: int = 2,
        N_iterations: int = 100,
        value_threshold: float = np.inf,
        stopping_criterion: Optional[Callable[[DataBuffer], bool]] = None,
        gae_lambda: float = 0.0,
        is_normalize_advantages: bool = True,
        device: str = "cpu",
        entropy_coeff: float = 0.0,
        policy_checkpoint_path: str = "",
        critic_checkpoint_path: str = "",
    ):
        if len(policy_checkpoint_path) != 0:
            policy_model.load_state_dict(torch.load(policy_checkpoint_path))

        if len(critic_checkpoint_path) != 0:
            critic_model.load_state_dict(torch.load(critic_checkpoint_path))
        
        super().__init__(policy_model, 
                         critic_model, 
                         sampling_time, 
                         running_objective, 
                         simulator, 
                         critic_n_epochs, 
                         policy_n_epochs, 
                         critic_opt_method_kwargs, 
                         policy_opt_method_kwargs, 
                         critic_opt_method, 
                         policy_opt_method, 
                         running_objective_type, 
                         critic_td_n, 
                         cliprange, 
                         discount_factor, 
                         observer, 
                         N_episodes, 
                         N_iterations, 
                         value_threshold, 
                         stopping_criterion, 
                         gae_lambda, 
                         is_normalize_advantages, 
                         device, 
                         entropy_coeff)


class MyREINFORCE(RLScenario):
    """Implements the REINFORCE algorithm."""

    def __init__(
        self,
        policy_model: PerceptronWithTruncatedNormalNoise,
        sampling_time: float,
        running_objective: RunningObjective,
        simulator: Simulator,
        policy_opt_method_kwargs: Dict[str, Any],
        policy_opt_method: Type[torch.optim.Optimizer] = torch.optim.Adam,
        policy_n_epochs: int = 1,
        discount_factor: float = 1.0,
        observer: Optional[Observer] = None,
        N_episodes: int = 4,
        N_iterations: int = 100,
        value_threshold: float = np.inf,
        is_with_baseline: bool = True,
        is_do_not_let_the_past_distract_you: bool = True,
        stopping_criterion: Optional[Callable[[DataBuffer], bool]] = None,
        checkpoint_path: str = ""
    ):
        """Initialize an REINFORCE object.

        Args:
            policy_model: The
                policy network model that defines the policy
                architecture.
            sampling_time: The time step between agent actions
                in the environment.
            running_objective: Function calculating
                the reward or cost at each time step when an action is
                taken.
            simulator: The environment in which the agent
                operates, providing state, observation.
            policy_opt_method_kwargs: The keyword
                arguments for the policy optimizer method.
            policy_opt_method (Type[torch.optim.Optimizer], optional):
                The policy optimizer method.
            n_epochs: The number of epochs used by the
                policy optimizer.
            discount_factor: The discount factor used
                by the RLScenario. Defaults to 1.0.
            observer: The observer object
                that estimates the state of the environment from observations.
            N_episodes: The number of episodes per iteration.
            N_iterations: The total number of iterations
                for training.
            is_with_baseline: Whether to use baseline as value (i.e. cumulative cost or reward)
                from previous iteration.
            is_do_not_let_the_past_distract_you: Whether to use tail total costs or not.

        Returns:
            None: None
        """
        if len(checkpoint_path) != 0:
            policy_model.load_state_dict(torch.load(checkpoint_path))
        super().__init__(
            **get_policy_gradient_kwargs(
                sampling_time=sampling_time,
                running_objective=running_objective,
                simulator=simulator,
                discount_factor=discount_factor,
                observer=observer,
                N_episodes=N_episodes,
                N_iterations=N_iterations,
                value_threshold=value_threshold,
                policy_type=PolicyReinforce,
                policy_model=policy_model,
                policy_n_epochs=policy_n_epochs,
                policy_opt_method=policy_opt_method,
                policy_opt_method_kwargs=policy_opt_method_kwargs,
                is_reinstantiate_policy_optimizer=False,
                policy_kwargs=dict(
                    is_with_baseline=is_with_baseline,
                    is_do_not_let_the_past_distract_you=is_do_not_let_the_past_distract_you,
                ),
                is_use_critic_as_policy_kwarg=False,
                stopping_criterion=stopping_criterion,
            ),
        )

    def run_episode(self, episode_counter, iteration_counter):
        self.episode_counter = episode_counter
        self.iteration_counter = iteration_counter
        while self.sim_status != "episode_ended":
            self.sim_status = self.step()
            
            if self.step_counter != 0 and self.step_counter % self.simulator.appox_num_step == 0:
                break


class MyCALF(CALF):
    def run_episode(self, episode_counter, iteration_counter):
        self.episode_counter = episode_counter
        self.iteration_counter = iteration_counter
        while self.sim_status != "episode_ended":
            self.sim_status = self.step()
            
            if self.step_counter != 0 and self.step_counter % self.simulator.appox_num_step == 0:
                break
