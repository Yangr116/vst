# Copyright 2025 [Visual Spatial Tuning] Authors
import os
import json
from glob import glob
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("results_dir", type=str)
args = parser.parse_args()


results_dir = args.results_dir
print(results_dir)

result_files = glob(os.path.join(results_dir, "*.json"))

combined_data = {}
for result_file in result_files:
    result = json.load(open(result_file))
    for task in result:
        task_id = task['task_id']
        # Check if the task_id is already a key in our combined_data dictionary.
        if task_id not in combined_data:
            # If it's the first time we see this task_id,
            # we add the entire task dictionary to our collection.
            combined_data[task_id] = task
        else:
            # If we have seen this task_id before, we need to merge.
            # We extend the existing lists with the new data.
            # The `+=` operator for lists is equivalent to the `extend()` method.
            combined_data[task_id]['episode_indices'] += task['episode_indices']
            combined_data[task_id]['task_successes'] += task['task_successes']

# cal success rate
success_rate_all = []
for task_id in combined_data:
    task = combined_data[task_id]
    success_rate = sum(task['task_successes']) / len(task['task_successes'])
    print(f"Task ID: {task_id}, Success Rate: {success_rate}")
    success_rate_all.append(success_rate)

# cal avg success rate
avg_success_rate = sum(success_rate_all) / len(success_rate_all)
print(f"Average Success Rate: {avg_success_rate}")
