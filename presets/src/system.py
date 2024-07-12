from regelum.utils import rg
from regelum.system import (
    ThreeWheeledRobotKinematic,
)
from regelum.animation import DefaultAnimation

from regelum.callback import detach, ThreeWheeledRobotAnimation
from regelum.system import System   

@ThreeWheeledRobotAnimation.attach
@DefaultAnimation.attach
@detach
class MyThreeWheeledRobotKinematic(ThreeWheeledRobotKinematic):
    """The parameters correspond to those of Robotis TurtleBot3."""

    action_bounds = [[-0.22, 0.22], [-2.84, 2.84]]