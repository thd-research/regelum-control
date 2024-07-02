import torch
from typing import Union, Dict, Any
from pathlib import Path
from regelum.callback import HistoricalCallback, ScenarioStepLogger
from regelum.scenario import RLScenario
from rich.logging import RichHandler


class HandlerChecker(ScenarioStepLogger):
    """A callback which allows to log every step of simulation in a scenario."""
    def is_target_event(self, obj, method, output, triggers):
        try:
            if len(self._metadata["logger"].handlers) == 0:
                self._metadata["logger"].addHandler(RichHandler())
        except Exception as err:
            print("Error:", err)


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