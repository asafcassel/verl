import os
import sys
import subprocess

def run_verl_generation(group_idx, ckpt_step):
    """
    Converts the bash script parameters to a Python subprocess call
    for the verl.trainer.main_generation script.
    """
    home = os.path.expanduser("~")

    data_path = os.path.join(home, "data", "deepscaler_percentiles", f"test_percentiles_{10*group_idx}-{10*(group_idx+1)}.parquet")
    save_path = os.path.join(home, "data", "deepscaler_percentiles_responses", f"ckpt_step_{ckpt_step}", f"test_percentiles_{10*group_idx}-{10*(group_idx+1)}.parquet")
    if ckpt_step == 0:
        model_path = "Qwen/Qwen2.5-Math-1.5B"
    else:
        model_path = os.path.join(home, "checkpoints", "deepscaler_percentiles", "Qwen2.5-Math-1.5B--deepscaler-percentiles-20-90_hedge", f"global_step_{ckpt_step}")

    command = [
        "python3",
        "-m",
        "verl.trainer.main_generation",
        "trainer.nnodes=1",
        "trainer.n_gpus_per_node=4",
        f"data.path={data_path}",
        "data.prompt_key=prompt",
        "data.n_samples=3",
        f"data.output_path={save_path}",
        f"model.path={model_path}",
        "+model.trust_remote_code=True",
        "rollout.temperature=1.0",
        "rollout.top_k=50",
        "rollout.top_p=0.7",
        "rollout.prompt_length=1024",
        "rollout.response_length=3000",
        "rollout.tensor_model_parallel_size=2",
        "rollout.gpu_memory_utilization=0.8"
    ]

    print(f"Running command: {' '.join(command)}")

    try:
        # This will execute the command and wait for it to complete.
        # set -x equivalent for Python's subprocess is implicitly handled by printing the command
        # before execution. For explicit stderr/stdout capture, you can use:
        # result = subprocess.run(command, check=True, capture_output=True, text=True)
        # print(result.stdout)
        # print(result.stderr)
        subprocess.run(command, check=True)
        print("verl.trainer.main_generation completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error running verl.trainer.main_generation: {e}")
        print(f"Return code: {e.returncode}")
        # if capture_output was used:
        # print(f"Stdout: {e.stdout}")
        # print(f"Stderr: {e.stderr}")
    except FileNotFoundError:
        print("Error: 'python3' command not found. Make sure Python is installed and in your PATH.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        ckpt_step = sys.argv[1]
    else:
        ckpt_step = 0
    for i in range(10):
        run_verl_generation(group_idx=i, ckpt_step=ckpt_step)