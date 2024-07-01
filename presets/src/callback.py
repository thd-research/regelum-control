import torch
from typing import Union
from pathlib import Path
from regelum.callback import HistoricalCallback
from regelum.scenario import RLScenario
from regelum.critic import Critic
from regelum.policy import Policy


class PolicyModelSaver(HistoricalCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iteration_counter = 1

    def is_target_event(self, obj, method, output, triggers):
        if (
            isinstance(obj, RLScenario)
            and method == "pre_optimize"
        ):
            which, event, time, episode_counter, iteration_counter = output
            return which == "Policy"

    def on_function_call(self, obj, method, outputs):
        save_model(
            self,
            torch_nn_module=obj.policy.model,
            iteration_counter=self.iteration_counter,
        )
        self.iteration_counter += 1


def save_model(
    cls: Union[PolicyModelSaver],
    torch_nn_module: torch.nn.Module,
    iteration_counter: int,
) -> None:
    torch.save(
        torch_nn_module.state_dict(),
        Path(".callbacks")
        / cls.__class__.__name__
        / f"model_it_{iteration_counter:05}",
    )