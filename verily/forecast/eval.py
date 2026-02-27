import argparse

import datasets
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from verily.forecast import analysis


def prepare_df_results(
    df: pd.DataFrame, dataset: datasets.arrow_dataset.Dataset = None, probs_agg="mean"
):
    # Extract probability aggregation from output sequences
    df["probability_scores"] = df["probability_scores"].apply(eval)
    agg_methods = {"mean": np.mean, "max": np.max, "median": np.median}
    df["probability_agg"] = df["probability_scores"].apply(lambda x: agg_methods[probs_agg](x))

    # Create age bucket column
    er_predictor = analysis.ERPredictor(None, None, None)
    age_labels = ["<18", "18-25", "25-35", "35-45", "45-55", "55-65", "65+"]
    df["age"] = df["original_text"].apply(er_predictor.extract_age)
    bins = [0, 18, 25, 35, 45, 55, 65, float("inf")]
    df["age_bucket"] = pd.cut(df["age"], bins=bins, labels=age_labels, right=False)

    if dataset is None:
        return df

    # Create has_pr column
    prs_data = {
        "subject_id": dataset["subject_id"],
        "has_prs": np.array(dataset["prs"]).any(axis=2).reshape(-1),
    }

    if len(df) != len(prs_data["subject_id"]):
        raise ValueError("Mismatch in length of DataFrame and prs_data.")

    df = df.merge(pd.DataFrame(prs_data), on="subject_id")
    return df


def bootstrap_auc(y_true, y_pred_probs, n_iterations=1000, confidence_level=0.95):
    bootstrap_aucs = []
    while len(bootstrap_aucs) < n_iterations:
        # Create a bootstrap sample by sampling indices with replacement
        n_size = len(y_true)
        indices = np.random.choice(n_size, size=n_size, replace=True)

        # Ensure there are both classes in the bootstrap sample
        if len(np.unique(y_true[indices])) < 2:
            # If not, skip this iteration as AUROC is not defined
            continue

        # Calculate AUROC for the bootstrap sample
        auc = roc_auc_score(y_true[indices], y_pred_probs[indices])
        bootstrap_aucs.append(auc)

    # 4. Calculate the confidence interval
    # Convert the list of AUCs to a NumPy array for easier percentile calculation
    bootstrap_aucs = np.array(bootstrap_aucs)

    # Calculate the lower and upper percentile bounds
    alpha = (1.0 - confidence_level) / 2.0
    lower_bound = np.percentile(bootstrap_aucs, alpha * 100)
    upper_bound = np.percentile(bootstrap_aucs, (1 - alpha) * 100)

    return lower_bound, upper_bound


def _print_auc_stats(data, group_name):
    if len(data) > 0 and len(np.unique(data["label_token"])) == 2:
        y_true_stratum = data["label_token"].values
        y_pred_probs_stratum = data["probability_agg"].values
        stratum_auc = roc_auc_score(y_true_stratum, y_pred_probs_stratum)
        lower_bound, upper_bound = bootstrap_auc(y_true_stratum, y_pred_probs_stratum)
        print(
            f"{group_name}: AUC={stratum_auc:.4f} (95% CI: {lower_bound:.4f}-{upper_bound:.4f}) (n={len(data)})"
        )


def main(args):
    # Load inference output file and original dataset if needed (for PRS)
    preds_df = pd.read_csv(args.inference_path)
    dataset = datasets.load_from_disk(args.dataset_path) if args.dataset_path else None

    # Preprocess data
    df = prepare_df_results(preds_df, dataset, probs_agg=args.probs_agg)

    # Perform evaluation
    y_true = df["label_token"].values
    y_pred_probs = df["probability_agg"].values

    # AUC
    overall_auc = roc_auc_score(y_true, y_pred_probs)
    lower_bound, upper_bound = bootstrap_auc(y_true, y_pred_probs)
    print(f"Overall AUC: {overall_auc:.4f} (95% CI: {lower_bound:.4f}-{upper_bound:.4f})")

    if args.age:
        print("\nAge-stratified Analysis:")
        print("-" * 30)
        age_buckets = sorted(df["age_bucket"].dropna().unique())
        for age_bucket in age_buckets:
            stratum_df = df[df["age_bucket"] == age_bucket]
            _print_auc_stats(stratum_df, f"Age {age_bucket}")

    if args.prs:
        if not args.dataset_path:
            raise ValueError("Dataset path must be specified if --prs is used.")
        print("\nPRS-stratified Analysis:")
        print("-" * 30)
        for has_prs in sorted(df["has_prs"].unique()):
            stratum_df = df[df["has_prs"] == has_prs]
            _print_auc_stats(stratum_df, f"PRS={has_prs}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inference-path", type=str, required=True, help="Path to the inference results CSV."
    )
    parser.add_argument(
        "--probs-agg",
        type=str,
        required=False,
        help="Aggregation method for probabilities",
        choices=["mean", "max", "median"],
        default="mean",
    )
    parser.add_argument(
        "--age",
        action="store_true",
        default=False,
        help="Stratify by age (10-year buckets)",
    )
    parser.add_argument(
        "--prs",
        action="store_true",
        default=False,
        help="Stratify by whether pt has PRS data",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=False,
        help="Path to the dataset directory. Required if --prs is specified.",
    )
    args = parser.parse_args()
    main(args)
