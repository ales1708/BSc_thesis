import numpy as np
import json
import os
from NARM_grouped import main

def evaluate_algorithm(group, evals, iterations, dataset_name, algo_name, subsampling_factor, use_random):
    runtimes = []
    confidences = []
    supports = []
    fitnesses = []
    lifts = []
    zhangs = []
    yulesqs = []
    coverages = []
    n_rules_learned = []

    for i in range(iterations):
        print(f"Group: {group}, Evals: {evals}, Iteration: {i}")
        runtime, rules = main(grouped=group, evaluations=evals, dataset_name=dataset_name, algo_name=algo_name, subsampling_factor=subsampling_factor, use_random=use_random)

        runtimes.append(runtime)
        if len(rules) == 0:
            confidences.append(0)
            supports.append(0)
            fitnesses.append(0)
            lifts.append(0)
            zhangs.append(0)
            yulesqs.append(0)
            coverages.append(0)
            n_rules_learned.append(0)

            print(f"Runtime: {runtime}, rules learned: {0}\n\n")
            continue

        confidences.append(rules.mean("confidence"))
        supports.append(rules.mean("support"))
        fitnesses.append(rules.mean("fitness"))
        lifts.append(rules.mean("lift"))
        zhangs.append(rules.mean("zhang"))
        yulesqs.append(rules.mean("yulesq"))
        coverages.append(rules.mean("coverage"))
        n_rules_learned.append(len(rules))

        print(f"Runtime: {runtime}, rules learned: {len(rules)}\n\n")

    return {
        "runtimes": runtimes,
        "confidences": confidences,
        "supports": supports,
        "fitnesses": fitnesses,
        "lifts": lifts,
        "zhangs": zhangs,
        "yulesqs": yulesqs,
        "coverages": coverages,
        "n_rules_learned": n_rules_learned,
        "mean_runtime": np.mean(runtimes),
        "mean_confidence": np.mean(confidences),
        "mean_support": np.mean(supports),
        "mean_fitness": np.mean(fitnesses),
        "mean_lift": np.mean(lifts),
        "mean_zhang": np.mean(zhangs),
        "mean_yulesQ": np.mean(yulesqs),
        "mean_coverage": np.mean(coverages),
        "mean_n_rules_learned": np.mean(n_rules_learned)
    }

def main_evaluation():
    results = []
    dataset_name = "lbnl_fdd"
    algo_name = "GWO"
    use_random = True
    subsampling_factor = 0.2

    for evals in [50000]:
        eval_results = []
        print(f"Current evals: {evals}, dataset: {dataset_name}, algo: {algo_name}")
        for group in [True, False]:
            iterations = 25
            print(f"Amount of iterations: {iterations}")
            result = evaluate_algorithm(group, evals, iterations, dataset_name, algo_name, subsampling_factor, use_random)
            results.append({
                "group": group,
                "evals": evals,
                "iterations": iterations,
                **result
            })
            eval_results.append({
                "group": group,
                "evals": evals,
                "iterations": iterations,
                **result
            })
            print("-------------------")
        if dataset_name == "leakdb":
            with open(f'results/{algo_name}/{dataset_name}/results_{evals}_sf({subsampling_factor}).json', 'w') as file:
                json.dump(eval_results, file, indent=4)
        elif dataset_name == "lbnl_fdd":
            if use_random:
                with open(f'results/{algo_name}/{dataset_name}/results_{evals}_rand.json', 'w') as file:
                    json.dump(eval_results, file, indent=4)
            else:
                with open(f'results/{algo_name}/{dataset_name}/results_{evals}_extra.json', 'w') as file:
                    json.dump(eval_results, file, indent=4)
        print("++++++++++++++++++++++++++++++++++", end="\n\n")

    return results

if __name__ == "__main__":
    print("Starting evaluation")
    results = main_evaluation()
    # Further processing of results can be done here

    print("Evaluation done")
    with open('new_results_all.json', 'w') as file:
        json.dump(results, file)
