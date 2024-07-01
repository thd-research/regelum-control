DATA_ROOT="/home/robosrv/huyhoang/regelum-control-thd/regelum_data/outputs"
CKPT_NAME=model_it_00328
CKPT_PATH="${DATA_ROOT}/2024-07-01/10-02-34__reinforce_3wrobot_kin/3/.callbacks/PolicyModelSaver/${CKPT_NAME}"

python run_stable.py \
    +seed=0 \
    --jobs=-1 \
    scenario=reinforce_load \
    system=3wrobot_kin \
    --experiment=reinforce_3wrobot_kin \
    scenario.N_episodes=1 \
        scenario.N_iterations=1 \
    scenario.policy_opt_method_kwargs.lr=0.01 \
    scenario.policy_model.n_hidden_layers=2 \
    scenario.policy_model.dim_hidden=15 \
    scenario/policy_model=perceptron_simple \
    scenario.policy_model.normalize_output_coef=0.1 \
    scenario.policy_model.std=0.15 \
    scenario.policy_model.leaky_relu_slope=0.01 \
    scenario.checkpoint_path=${CKPT_PATH} \
    simulator.time_final=10 \
    --interactive --fps=10
    