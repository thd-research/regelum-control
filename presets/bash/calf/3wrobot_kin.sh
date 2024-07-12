python3.10 run_stable.py \
    +seed=1 \
    scenario=calf \
    scenario.N_iterations=30 \
    system=3wrobot_kin_ros \
    initial_conditions=ic_3wrobot_kin \
    running_objective=quadratic_3wrobot_kin \
    scenario.critic_model.quad_matrix_type=full \
    --experiment=calf_3wrobot_kin_test \
    +scenario.critic_safe_decay_param=0.001 \
    +scenario.critic_lb_parameter=1.0E-1 \
    +scenario.critic_regularization_param=3000 \
    +scenario.critic_learning_norm_threshold=1 \
    scenario.critic_td_n=10 \
    scenario.critic_batch_size=32 \
    scenario.N_iterations=100 \
    +scenario.critic_model.add_random_init_noise=True \
    system_specific.sampling_time=0.1 \
        nominal_policy=nominal_3wrobot_kin_ros \
    simulator.time_final=30 \
    --interactive