# encoding: utf-8
# usage example: python ~/verl/examples/generation/convert_ckpt_to_huggingface.py ~/checkpoints/deepscaler_percentiles/Qwen2.5-Math-1.5B--deepscaler-percentiles-percentiles_20-90_lr_4x/global_step_1704/actor/ Qwen/Qwen2.5-Math-1.5B ~/checkpoints/Huggingface_Qwen2.5-Math-1.5B--deepscaler-percentiles-percentiles_20-90_lr_4x_step_1704
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch
import fire
from glob import glob
from collections import defaultdict


def main(fsdp_checkpoint_path, huggingface_model_path, output_path):
    state_dict = defaultdict(list)

    world_size = 4
    for rank in range(world_size):
        filepath = f"{fsdp_checkpoint_path}/model_world_size_{world_size}_rank_{rank}.pt"
        print('loading', filepath)
        this_state_dict = torch.load(filepath, weights_only=False)
        for key, value in this_state_dict.items():
            state_dict[key].append(value.to_local())

    for key in state_dict:
        state_dict[key] = torch.cat(state_dict[key], dim=0)

    config = AutoConfig.from_pretrained(huggingface_model_path)
    model = AutoModelForCausalLM.from_config(config)
    model.load_state_dict(state_dict)

    #for filepath in glob(f'{fsdp_checkpoint_path}/model_*.pt'):
    #    part_state_dict = torch.load(filepath)
    #    model.load_state_dict(part_state_dict)

    model.save_pretrained(output_path, max_shard_size="10GB")

    tokenizer = AutoTokenizer.from_pretrained(huggingface_model_path)
    tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    fire.Fire(main)