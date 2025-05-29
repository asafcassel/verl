import os
import sys

custom_sampler = 'hedge'
num_groups = 10
train_start_group = 2
train_end_group = 8
b = 100 // num_groups
base_train_path = "verl/verl/data/deepscaler_percentiles/train_percentiles"

# Determine task_type and train_files based on command-line arguments
if len(sys.argv) == 1:
    task_type = f"{b * train_start_group}-{b * (train_end_group + 1)}"
    train_files = [f"{base_train_path}_{b*i}-{b*(i+1)}.parquet" for i in range(train_start_group, train_end_group + 1)]
else:
    task_type = sys.argv[1]
    train_files = [f"{base_train_path}_{task_type}.parquet"]

# Define test files
base_test_path = "verl/verl/data/deepscaler_percentiles/test_percentiles"
test_files = [f"{base_test_path}_{b*i}-{b*(i+1)}.parquet" for i in range(num_groups)]

custom_sampler_config = []
if custom_sampler is not None:
    task_type = f"{task_type}_{custom_sampler}"
    if custom_sampler == 'hedge':
        custom_sampler_config = [
            "data.hedge.enable=enable",
            "data.hedge.init_type=softmax",
            "data.hedge.replacement=True",
            "data.hedge.eta=0.02",
            "data.hedge.gamma=0.01",
        ]
    elif custom_sampler == 'adarft':
        custom_sampler_config = [
            "data.adarft.enable=enable",
            "data.adarft.beta=0.5",
            "data.adarft.alpha=2",
            "data.adarft.eta=3",
            f"data.adarft.d_min={100 - (b * (train_end_group + 1))}",
            f"data.adarft.d_max={100 - (b * train_start_group)}",
        ]
    else:
        raise f'Invalid custom sampler {custom_sampler}'
    
# Construct the command for the Python script
command = [
    "python3",
    "-B",
    "-m",
    "verl.trainer.main_ppo",
    f"data.train_files={train_files}",
    f"data.val_files={test_files}",
    "data.train_batch_size=1024",
    "data.max_prompt_length=1024",
    "data.max_response_length=3000",
    *custom_sampler_config,
    "data.truncation='left'",
    "actor_rollout_ref.model.path=Qwen/Qwen2.5-Math-1.5B",
    "actor_rollout_ref.actor.optim.lr=4e-6",
    "actor_rollout_ref.model.use_remove_padding=True",
    "actor_rollout_ref.actor.ulysses_sequence_parallel_size=1",
    "actor_rollout_ref.model.enable_gradient_checkpointing=True",
    "actor_rollout_ref.actor.ppo_mini_batch_size=1024",
    "actor_rollout_ref.actor.use_dynamic_bsz=True",
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16000",
    "actor_rollout_ref.actor.fsdp_config.param_offload=False",
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
    "actor_rollout_ref.rollout.tensor_model_parallel_size=2",
    "actor_rollout_ref.rollout.name=vllm",
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.5",
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16000",
    "actor_rollout_ref.ref.fsdp_config.param_offload=True",
    "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16000",
    "actor_rollout_ref.rollout.n=1",
    "critic.optim.lr=4e-5",
    "critic.ulysses_sequence_parallel_size=1",
    "critic.model.use_remove_padding=True",
    "critic.model.path=Qwen/Qwen2.5-Math-1.5B",
    "critic.model.enable_gradient_checkpointing=True",
    "critic.ppo_max_token_len_per_gpu=20000",
    "critic.model.fsdp_config.param_offload=False",
    "critic.model.fsdp_config.optimizer_offload=False",
    "algorithm.kl_ctrl.kl_coef=0.001",
    "trainer.critic_warmup=0",
    "trainer.logger=['console','wandb']",
    "trainer.project_name='deepscaler_percentiles'",
    f"trainer.experiment_name='Qwen2.5-Math-1.5B--deepscaler-percentiles-{task_type}'",
    "trainer.n_gpus_per_node=4",
    "trainer.nnodes=1",
    "trainer.save_freq=400",
    "trainer.test_freq=20",
    f"trainer.total_epochs={500 // len(train_files)}",
]

# Add any additional arguments passed to the Python script
if len(sys.argv) > 2:
    command.extend(sys.argv[2:])

# Print and execute the command
print(f"Executing command: {' '.join(command)}")
os.execvp(command[0], command)