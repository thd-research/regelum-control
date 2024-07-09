# # Checkpoint 1
# DATA_ROOT="/regelum-control/regelum_data/outputs/2024-07-05/22-01-10__reinforce_3wrobot_kin_ros/0"
# CKPT_NAME=model_it_00040

#  Checkpoint 2 From Checkpoint 1
# DATA_ROOT="/regelum-control/regelum_data/outputs/2024-07-06/09-23-30__reinforce_3wrobot_kin_ros/0" 
# CKPT_NAME=model_it_00010

# Checkpoint 3
DATA_ROOT="/regelum-control/regelum_data/outputs/2024-07-08/21-25-58__reinforce_3wrobot_kin_ros/0"
CKPT_NAME=model_it_00007

CKPT_PATH="${DATA_ROOT}/.callbacks/PolicyModelSaver/${CKPT_NAME}"

IS_VALIDATION="true"
# CKPT_PATH=""

python3.10 run_stable.py \
            +seed=6 \
            --jobs=-1 \
            simulator=ros \
            scenario=reinforce_train \
            initial_conditions=ic_3wrobot_kin \
            running_objective=quadratic_3wrobot_kin \
            system=3wrobot_kin_ros \
            --experiment=reinforce_3wrobot_kin_ros \
            scenario.N_episodes=1 \
                scenario.N_iterations=1 \
            scenario.policy_opt_method_kwargs.lr=0.01   \
            scenario.policy_model.n_hidden_layers=2 \
            scenario.policy_model.dim_hidden=30 \
            scenario/policy_model=perceptron_simple \
            scenario.policy_model.normalize_output_coef=1 \
            scenario.policy_model.std=0.01 \
            scenario.policy_model.leaky_relu_slope=0.1 \
            simulator.time_final=100 \
            system_specific.sampling_time=0.1 \
            scenario.checkpoint_path=${CKPT_PATH} \
            simulator.stop_if_reach_target=${IS_VALIDATION}
