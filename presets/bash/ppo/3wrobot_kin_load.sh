DATA_ROOT="/regelum-control/regelum_data/outputs"
CKPT_NAME=model_it_00072
CKPT_PATH="${DATA_ROOT}/2024-08-07/18-32-26__ppo_3wrobot_kin/0/.callbacks"


python3.10 run_stable.py \
          +seed=7 \
          scenario=ppo_load \
          system=3wrobot_kin \
          --experiment=ppo_3wrobot_kin_stored_weights \
          scenario.N_episodes=1 \
          scenario.N_iterations=1 \
          scenario.policy_n_epochs=30 \
          scenario.critic_n_epochs=30 \
          scenario.policy_opt_method_kwargs.lr=0.0005 \
          scenario.policy_model.n_hidden_layers=2 \
          scenario.policy_model.dim_hidden=15 \
          scenario.policy_model.std=0.1 \
          scenario.critic_model.n_hidden_layers=2 \
          scenario.critic_model.dim_hidden=15 \
          scenario.critic_opt_method_kwargs.lr=0.1 \
          scenario.gae_lambda=0.96 \
          simulator.time_final=15 \
          scenario.policy_checkpoint_path="${CKPT_PATH}/PolicyModelSaver/${CKPT_NAME}" \
          scenario.critic_checkpoint_path="${CKPT_PATH}/CriticModelSaver/${CKPT_NAME}" \
          --interactive
