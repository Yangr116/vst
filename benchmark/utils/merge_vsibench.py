# Copyright 2025 [Visual Spatial Tuning] Authors
import os
import json
import argparse
import pandas as pd
from glob import glob
from loguru import logger as eval_logger
from collections import defaultdict
import pandas as pd


MCA_QUESTION_TYPES = [
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
]
NA_QUESTION_TYPES = [
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
]

METRICS_FOR_MCA = {
    "accuracy": "exact_match",
}

METRICS_FOR_NA = {
    "MRA:.5:.95:.05": "partial(mean_relative_accuracy, start=.5, end=.95, interval=.05)",
}


def vsibench_aggregate_results(results):
    results = pd.DataFrame(results)
    
    output = {}

    for question_type, question_type_indexes in results.groupby('question_type').groups.items():
        per_question_type = results.iloc[question_type_indexes]
        
        if question_type in MCA_QUESTION_TYPES:
            for metric in METRICS_FOR_MCA.keys():
                output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
        elif question_type in NA_QUESTION_TYPES:
            for metric in METRICS_FOR_NA.keys():
                if metric == 'success_rate':
                    output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
                else:
                    output[f"{question_type}_{metric}"] = per_question_type[metric].mean()

        else:
            raise ValueError(f"Unknown question type: {question_type}")

    output['object_rel_direction_accuracy'] = sum([
        output.pop('object_rel_direction_easy_accuracy'),
        output.pop('object_rel_direction_medium_accuracy'),
        output.pop('object_rel_direction_hard_accuracy'),
    ]) / 3.
    
    output['overall'] = sum([_ for _ in output.values()]) / len(output)
    output = {k: v*100. for k, v in output.items()}
    eval_logger.info(f"Evaluation results: {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True, type=str)
    args = parser.parse_args()

    # merge all jsonfile in the subfolder
    jsonfiles = glob(args.result_dir + "/*.json")
    merged_results = defaultdict(list)
    for jsonfile in jsonfiles:
        data = json.load(open(jsonfile, 'r'))
        for metric, items in data.items():
            merged_results[metric].extend(items)

    for metric, items in merged_results.items():
        output = vsibench_aggregate_results(items)
        df = pd.DataFrame([output])
        df.to_csv(os.path.join(args.result_dir, 'VSI_Bench_acc.csv'), index=False)
        eval_logger.info(output)
