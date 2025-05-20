# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the MATH-lighteval dataset to parquet format
"""

import os
import datasets


from verl.utils.hdfs_io import copy, makedirs
import argparse


def split_to_percentiles(ds, num_groups):
    # Sort the dataset by 'solved_percentage'
    ds_sorted = ds.sort("solved_percentage")

    # Calculate the size of each group
    group_size = len(ds_sorted['train']) // num_groups

    # Initialize a list to store the grouped and split datasets
    grouped_datasets = []

    for i in range(num_groups):
        # Get the data for the current group
        start_index = i * group_size
        end_index = start_index + group_size
        group_data = ds_sorted['train'].select(range(start_index, end_index))

        # Shuffle the current group
        group_data_shuffled = group_data.shuffle(seed=42) # Using a fixed seed for reproducibility

        # Split the shuffled group into train and test sets
        # The test set is 10% of the total group sample size
        test_size = int(len(group_data_shuffled) * 0.1)
        train_test_split = group_data_shuffled.train_test_split(test_size=test_size)

        # Store the split datasets for the current group
        grouped_datasets.append(train_test_split)
    return grouped_datasets


if __name__ == '__main__':
    num_groups = 5
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data/deepscaler_percentiles')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    data_source = '"lime-nlp/DeepScaleR_Difficulty", "Difficulty Score"'
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset("lime-nlp/DeepScaleR_Difficulty", "Difficulty Score", trust_remote_code=True)
    grouped_datasets = split_to_percentiles(dataset, num_groups=num_groups)

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."
    
    b = 100 // num_groups
    for i, group_data in enumerate(grouped_datasets):
        data_source_name = f'deepscaler_percentiles_{b*i}-{b*(i+1)}'
        # print(data_source_name)
        # add a row to each data item that represents a unique id
        def make_map_fn(split):

            def process_fn(example, idx):
                question = example.pop('problem')

                question = question + ' ' + instruction_following

                data = {
                    "data_source": data_source_name,
                    "prompt": [{
                        "role": "user",
                        "content": question
                    }],
                    "ability": "math",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": example.pop('ground_truth')
                    },
                    "extra_info": {
                        'difficulty': example.pop('solved_percentage'),
                        'index': idx,
                        'split': split,
                    }
                }
                # print(data['data_source'])
                return data

            return process_fn
        
        train_dataset = group_data['train']
        test_dataset = group_data['test']
        train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
        test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

        local_dir = args.local_dir
        hdfs_dir = args.hdfs_dir

        train_dataset.to_parquet(os.path.join(local_dir, f'train_percentiles_{b*i}-{b*(i+1)}.parquet'))
        test_dataset.to_parquet(os.path.join(local_dir, f'test_percentiles_{b*i}-{b*(i+1)}.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir, exist_ok=True)

        copy(src=local_dir, dst=hdfs_dir)
