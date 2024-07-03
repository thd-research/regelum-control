python3.10 run_stable.py \
            +seed=0 \
            --jobs=-1 \
            simulator=ros \
            scenario=reinforce_train \
            system=3wrobot_kin \
            --experiment=reinforce_3wrobot_kin \
            scenario.N_episodes=1 \
                scenario.N_iterations=3000 \
            scenario.policy_opt_method_kwargs.lr=0.01 \
            scenario.policy_model.n_hidden_layers=2 \
            scenario.policy_model.dim_hidden=15 \
            scenario/policy_model=perceptron_simple \
            scenario.policy_model.normalize_output_coef=0.1 \
            scenario.policy_model.std=0.15 \
            scenario.policy_model.leaky_relu_slope=0.01 \
            simulator.time_final=15 \
            system_specific.sampling_time=0.05
