set -x

if [ "$#" -eq 0 ]; then
    task_type=skew_difficult
else
    task_type="${1}"
fi

gsm8k_test_path=verl/verl/data/gsm8k/test.parquet
deepscaler_skew_difficult_train_path='verl/verl/data/deepscaler_'${task_type}'_train.parquet'

train_files="['$deepscaler_skew_difficult_train_path']"
test_files="['$gsm8k_test_path']"

# export VLLM_ATTENTION_BACKEND=XFORMERS
python3 -B -m verl.trainer.main_ppo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=3000 \
    data.truncation='left' \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-Math-1.5B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=1024 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16000 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16000 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16000 \
    actor_rollout_ref.rollout.n=1 \
    critic.ulysses_sequence_parallel_size=1 \
    critic.optim.lr=1e-5 \
    critic.ulysses_sequence_parallel_size=1 \
    critic.model.use_remove_padding=True \
    critic.model.path=Qwen/Qwen2.5-Math-1.5B \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_max_token_len_per_gpu=20000 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_examples' \
    trainer.experiment_name='Qwen2.5-Math-1.5B--deepscaler-'${task_type} \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=12 ${@:2}
