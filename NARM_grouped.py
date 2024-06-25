import numpy as np
import pandas as pd
import json

from niaarm import Dataset

from algos.de import DifferentialEvolution
from algos.hho import HarrisHawksOptimization
from algos.gwo import GreyWolfOptimizer
from algos.bat import BatAlgorithm
from algos.sca import SineCosineAlgorithm

from utils.Mine import get_rules


def subsample(data, group_info, percentage):
    total_groups = len(group_info)
    n = int(total_groups * percentage)

    selected_groups = np.random.choice(group_info, n, replace=False)

    # Extract the relevant features for the selected groups
    selected_features = []
    seen_features = set()
    for group in selected_groups:
        for feature in group.keys():
            if feature not in seen_features:
                selected_features.append(feature)
                seen_features.add(feature)

    # Filter the DataFrame to include only the selected features using .loc
    # Maintain the original order of columns
    filtered_df = data.loc[:, [col for col in data.columns if col in selected_features]]

    return filtered_df, selected_groups


def main(grouped, evaluations, algo_name, dataset_name, subsampling_factor=0.15):
    df = pd.read_csv(f'/datasets/water pipes/{dataset_name}.csv')
    with open(f"/datasets/water pipes/{dataset_name}_groups.json", "r") as f:
        group_info = json.load(f)

    if dataset_name == "leakdb":
        grouped_data, grouping_data = subsample(df, group_info, subsampling_factor)
        data = Dataset(grouped_data)
    elif dataset_name == "lbnl_fdd":
        data = Dataset(df)
        grouping_data = group_info
    else:
        raise ValueError("Invalid dataset name")

    metrics = ("support", "confidence")

    if algo_name == "DE":
        algo = DifferentialEvolution(
            population_size=50, differential_weight=0.5, crossover_probability=0.9, grouping=grouped
        )
    elif algo_name == "HHO":
        algo = HarrisHawksOptimization(grouping=grouped)
    elif algo_name == "GWO":
        algo = GreyWolfOptimizer(grouping=grouped)
    elif algo_name == "BAT":
        algo = BatAlgorithm(grouping=grouped)
    elif algo_name == "SCA":
        algo = SineCosineAlgorithm(grouping=grouped)
    else:
        raise ValueError("Invalid algorithm name")


    res = get_rules(data, algo, grouping_data, metrics, max_evals=evaluations, logging=False, grouping=True)

    run_time = res.run_time
    rules = res.rules

    return run_time, rules

if __name__ == "__main__":
    max_evals = 10000
    algo_name = "BAT"
    dataset_name = "lbnl_fdd"
    grouped = True

    run_time, rules = main(grouped, max_evals, algo_name, dataset_name)
    print(run_time)
    print(rules)