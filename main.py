import os
import argparse
import random
import numpy as np
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from deap import gp

from MTGPSC import SymbolicRegressorGP


# === 1. 数据集名单与路径 ===
ALL_DATASETS = [
    'andro', 'atp1d', 'atp7d', 'edm', 'enb', 'jura',
    'oes10', 'oes97', 'osales', 'scpf', 'slump', 'wq'
]


def dataset_path(dataset_key):
    return f"./data/{dataset_key}.arff"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def gp_predict(ind, pset, X):
    func = gp.compile(ind, pset)
    preds = []
    for x in X:
        try:
            preds.append(func(*x))
        except:
            preds.append(np.nan)
    return np.array(preds)


# === 2. arff数据加载&预处理 ===
def load_dataset_arff(path, n_outputs=None):
    with open(path, 'r') as f:
        arff_data = arff.loadarff(f)

    data_raw = arff_data[0]
    meta = arff_data[1]

    data = np.array(data_raw.tolist(), dtype=np.float64)
    attr_names = meta.names()

    # 自动推断输出列数
    # 你这里最好手动提供映射，因为不同数据集输出维度论文里是已知的
    if n_outputs is None:
        dataset_name = os.path.basename(path).replace(".arff", "").lower()
        output_dims = {
            'andro': 6,
            'atp1d': 6,
            'atp7d': 6,
            'edm': 2,
            'enb': 2,
            'jura': 3,
            'oes10': 16,
            'oes97': 16,
            'osales': 12,
            'scpf': 3,
            'slump': 3,
            'wq': 14,
        }
        if dataset_name not in output_dims:
            raise ValueError(f"Unknown dataset `{dataset_name}`, please specify n_outputs.")
        n_outputs = output_dims[dataset_name]

    X = data[:, :-n_outputs]
    Y = data[:, -n_outputs:]

    # 按论文：缺失值用样本均值填充
    # X
    if np.isnan(X).any():
        col_mean = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_mean, inds[1])

    # Y 一般不应该缺失，但保险起见也处理
    if np.isnan(Y).any():
        col_mean = np.nanmean(Y, axis=0)
        inds = np.where(np.isnan(Y))
        Y[inds] = np.take(col_mean, inds[1])

    return X, Y, attr_names


def average_nrmse(y_true, y_pred):
    vals = []
    for i in range(y_true.shape[1]):
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        denom = y_true[:, i].max() - y_true[:, i].min()
        if denom == 0:
            denom = 1.0
        vals.append(rmse / denom)
    return float(np.mean(vals))


def predict_multioutput(model, X):
    preds = []
    for i in range(len(model.final_model)):
        ind = model.final_model[i]
        y_pred_i = gp_predict(ind, model.pset, X)
        preds.append(y_pred_i)
    return np.array(preds).T


def run_one_dataset_once(dataset_key, run_seed, args):
    set_seed(run_seed)

    path = dataset_path(dataset_key)
    X, Y, attr_names = load_dataset_arff(path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=run_seed
    )

    model = SymbolicRegressorGP(
        n_generations=args.generations,
        pop_size=args.pop_size,
        crossover_prob=args.cxpb,
        mutation_prob=args.mutpb,
        verbose=args.verbose,
        tr=args.tr,
        semantic_crossover=True,

        use_llm=(not args.no_llm),
        llm_start_gen=args.llm_start_gen,
        llm_ratio=args.llm_ratio,
        llm_model=args.model,
        llm_api_base=args.api_base,
        llm_api_key=args.api_key,
        llm_temperature=args.temperature,
        llm_max_retries=args.llm_max_retries,
        llm_debug=args.llm_debug,
        llm_log_file=args.llm_log_file,
        llm_retry_on_fail=args.llm_retry_on_fail,
        llm_explore_min_semantic_dist=args.llm_explore_min_semantic_dist,
        llm_stagnation_window=args.llm_stagnation_window,
        llm_stagnation_epsilon=args.llm_stagnation_epsilon,
        llm_low_diversity_epsilon=args.llm_low_diversity_epsilon,
        llm_refine_top_k=args.llm_refine_top_k,
    )

    model.fit(X_train, y_train)

    y_train_pred = predict_multioutput(model, X_train)
    y_test_pred = predict_multioutput(model, X_test)

    train_anrmse = average_nrmse(y_train, y_train_pred)
    test_anrmse = average_nrmse(y_test, y_test_pred)

    return {
        "dataset": dataset_key,
        "seed": run_seed,
        "train_anrmse": train_anrmse,
        "test_anrmse": test_anrmse,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run MTGPSC / MTGPSC+LLM on ARFF datasets")

    parser.add_argument(
        "--dataset",
        nargs="+",
        required=True,
        help="Dataset name(s), e.g. andro atp1d, or 'all'"
    )
    parser.add_argument("--runs", type=int, default=30, help="Number of independent runs")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Explicit random seeds for runs, e.g. --seeds 0 7 42 99"
    )

    parser.add_argument("--api-key", type=str, default=None, help="LLM API key")
    parser.add_argument("--api-base", type=str, default="https://api.deepseek.com/v1", help="LLM API base URL")
    parser.add_argument("--model", type=str, default="deepseek-chat", help="LLM model name")

    parser.add_argument("--no-llm", action="store_true", help="Disable LLM augmentation")

    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--cxpb", type=float, default=0.9)
    parser.add_argument("--mutpb", type=float, default=0.1)
    parser.add_argument("--tr", type=float, default=0.3, help="random mating probability / transfer ratio")

    parser.add_argument("--llm-start-gen", type=int, default=50)
    parser.add_argument("--llm-ratio", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--llm-max-retries", type=int, default=3)

    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--llm-debug", action="store_true", help="Print LLM generation details")
    parser.add_argument("--llm-log-file", type=str, default=None, help="Save LLM logs to file")
    parser.add_argument("--llm-retry-on-fail", type=int, default=1)
    parser.add_argument("--llm-explore-min-semantic-dist", type=float, default=0.05)
    parser.add_argument("--llm-stagnation-window", type=int, default=5)
    parser.add_argument("--llm-stagnation-epsilon", type=float, default=1e-4)
    parser.add_argument("--llm-low-diversity-epsilon", type=float, default=1e-4)
    parser.add_argument("--llm-refine-top-k", type=int, default=10)
    return parser.parse_args()


def resolve_datasets(dataset_args):
    if len(dataset_args) == 1 and dataset_args[0].lower() == "all":
        return ALL_DATASETS

    dataset_args = [d.lower() for d in dataset_args]
    for d in dataset_args:
        if d not in ALL_DATASETS:
            raise ValueError(f"Unknown dataset: {d}. Available: {ALL_DATASETS}")
    return dataset_args


def resolve_run_seeds(args):
    if args.seeds is not None:
        if len(args.seeds) == 0:
            raise ValueError("--seeds is provided but empty.")
        return args.seeds
    return list(range(args.runs))


def main():
    args = parse_args()
    datasets = resolve_datasets(args.dataset)
    run_seeds = resolve_run_seeds(args)

    if (not args.no_llm) and (not args.api_key):
        raise ValueError("LLM is enabled, but --api-key is not provided.")

    all_results = []

    for dataset_key in datasets:
        print(f"\n{'=' * 80}")
        print(f"Dataset: {dataset_key}")
        print(f"{'=' * 80}")

        dataset_train_scores = []
        dataset_test_scores = []

        for run_idx, run_seed in enumerate(run_seeds):
            print(f"\n[Dataset={dataset_key}] Run {run_idx + 1}/{len(run_seeds)} | Seed={run_seed}")
            result = run_one_dataset_once(dataset_key, run_seed, args)
            all_results.append(result)

            dataset_train_scores.append(result["train_anrmse"])
            dataset_test_scores.append(result["test_anrmse"])

            print(
                f"Seed={run_seed} | "
                f"Train aNRMSE={result['train_anrmse']:.6f} | "
                f"Test aNRMSE={result['test_anrmse']:.6f}"
            )

        print(f"\nSummary for {dataset_key}:")
        print(
            f"Train mean ± std: {np.mean(dataset_train_scores):.6f} ± {np.std(dataset_train_scores):.6f}"
        )
        print(
            f"Test  mean ± std: {np.mean(dataset_test_scores):.6f} ± {np.std(dataset_test_scores):.6f}"
        )

    print(f"\n{'=' * 80}")
    print("All done.")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()