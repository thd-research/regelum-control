import sys, os

# ["2tank", "3wrobot", "3wrobot_ni", "cartpole", "inv_pendulum", "kin_point", "lunar_lander"])
# ["ddpg", "ddqn", "dqn", "dqn", "mpc", "pg", "pid", "rpo", "rpo_deep", "rql", "sarsa", "sdpg", "sql"]


class TestSetup:
    def __init__(self, config_path=None, config_name=None, **params):
        self.config_path = (
            config_path if config_path is not None else self.config_path_default
        )
        self.config_path = (
            os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
            + f"/../../presets/{self.config_path}"
        )
        self.config_name = (
            config_name if config_name is not None else self.config_name_default
        )
        self.params = params

    @property
    def config_path_default(self):
        return "stable-presets"
        # return os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + "/../../presets/general")

    @property
    def config_name_default(self):
        return "main"

    def _pre_setup(self):
        os.environ["HYDRA_FULL_ERROR"] = "1"
        sys.argv.insert(1, "--single-thread")
        sys.argv.insert(1, "--no-git")
        sys.argv.insert(1, "disallow_uncommitted=False")
        sys.argv.insert(1, "simulator.time_final=1")
        sys.argv.insert(1, "controller.sampling_time=0.5")
        sys.argv.insert(1, "scenario.N_episodes=2")
        sys.argv.insert(1, "scenario.N_iterations=1")
        sys.argv.insert(1, "+disallow_uncommitted=False")
        sys.argv.insert(1, "--experiment=TESTS")

    def __call__(self):
        sys.argv = [sys.argv[0]]
        for param in self.params:
            sys.argv.insert(1, f"{param}={self.params[param]}")
        self._pre_setup()
        os.environ[
            "REGELUM_RECENT_TEST_INFO"
        ] = f"Command line arguments: {' '.join(sys.argv[1:])}\nConfig: {os.path.abspath(self.config_path)}/{self.config_name}.yaml"
        return {"config_path": self.config_path, "config_name": self.config_name}

    def __str__(self):
        return f"{self.params['system']}_{self.params['controller']}"


class MPCTest(TestSetup):
    def __init__(self, system):
        super().__init__(
            system=system,
            controller="mpc",
            **{"controller.actor.predictor.prediction_horizon": 2},
        )


#######################################################################################################################
#######################################################################################################################

basic = [
    TestSetup(system="3wrobot_ni", controller="sdpg", **{"simulator.time_final": 3.0}),
    TestSetup(system="3wrobot_ni", controller="ppo", **{"simulator.time_final": 3.0}),
    TestSetup(system="3wrobot_ni", controller="ddpg"),
    TestSetup(system="3wrobot_ni", controller="reinforce"),
    TestSetup(system="3wrobot_ni", controller="dqn", **{"simulator.time_final": 0.5}),
    TestSetup(system="3wrobot_ni", controller="sarsa", **{"simulator.time_final": 0.5}),
    TestSetup(system="3wrobot_ni", controller="rpo", **{"simulator.time_final": 0.5}),
    TestSetup(
        system="3wrobot_ni", controller="rpo_torch", **{"simulator.time_final": 0.5}
    ),
    TestSetup(
        system="3wrobot_ni", controller="mpc_torch", **{"simulator.time_final": 0.5}
    ),
    TestSetup(system="3wrobot_ni", controller="mpc", **{"simulator.time_final": 0.5}),
]

"""
basic = [MPCTest(system="2tank"),
         TestSetup(system="3wrobot", controller="rpo"),
         TestSetup(system="3wrobot_ni", controller="rpo"),
         TestSetup(system="cartpole", controller="rql"),
         TestSetup(system="inv_pendulum", controller="rpo"),
         MPCTest(system="kin_point"),
         MPCTest(system="lunar_lander")]

extended = basic + \
           [TestSetup(system="kin_point", controller="ddpg"),
            TestSetup(system="kin_point", controller="sarsa")]
"""
