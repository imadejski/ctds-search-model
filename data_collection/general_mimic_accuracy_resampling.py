import argparse
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import norm


def read_data(cosine_path):
    """
    Reads in csvs with cosine similarity scores, ground-truth labels, and split
    """
    cosine_df = pd.read_csv(cosine_path)
    labels_df = pd.read_csv("/opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv")
    split_df = pd.read_csv("/opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-split.csv")

    return cosine_df, labels_df, split_df


def filter_split_type_data(split_df, labels_df, split_type):
    """
    Filters split df so that only specified split type exists
    """
    split_type_df = split_df[split_df["split"] == split_type]
    split_type_study_ids = split_type_df["study_id"].unique()
    split_type_labels_df = labels_df[labels_df["study_id"].isin(split_type_study_ids)]

    return split_type_labels_df


def transform_cosine_df(cosine_df, labels, embedding_types):
    """
    Transforms cosine similarity df for processing by accuracy function
    """
    data = []
    for label in labels:
        for emb in embedding_types:
            column_name = f"{label} {emb}"
            if column_name in cosine_df.columns:
                temp_df = cosine_df[
                    ["subject_id", "study_id", "dicom_id", column_name]
                ].copy()
                temp_df.rename(columns={column_name: "cosine_similarity"}, inplace=True)
                temp_df["label"] = label
                temp_df["embedding_type"] = emb

                temp_df["cosine_similarity"] = pd.to_numeric(
                    temp_df["cosine_similarity"], errors="coerce"
                )

                data.append(temp_df)
            else:
                raise ValueError(f"Column {column_name} does not exist in cosine_df")
    combined_df = pd.concat(data, ignore_index=True)

    return combined_df


def count_positive_cases(labels, split_type_labels_df):
    """
    Finds total number of positive cases, represented by 1, in df w labels
    """
    n_counts = {}
    for label in labels:
        label_positives = split_type_labels_df[split_type_labels_df[label] == 1]
        unique_positive_study_ids = label_positives["study_id"].unique()
        n_counts[label] = len(unique_positive_study_ids)

    return n_counts


def aggregate_cosine(df, method: Literal["max", "mean"]):
    """
    Aggregates cosine similarity scores by taking the max or average
    """
    if method == "max":
        aggregated_df = (
            df.groupby(["study_id", "label", "embedding_type"])["cosine_similarity"]
            .max()
            .reset_index()
        )
    elif method == "mean":
        aggregated_df = (
            df.groupby(["study_id", "label", "embedding_type"])["cosine_similarity"]
            .mean()
            .reset_index()
        )
    else:
        raise ValueError("Aggregation method must be 'max' or 'mean'")

    # Debugging: Print the size of the aggregated DataFrame
    print(f"Aggregated DataFrame size for method '{method}': {aggregated_df.shape}")
    return aggregated_df


def calculate_accuracy_and_top_k(
    labels,
    n_counts,
    cosine_max,
    cosine_mean,
    split_type_labels_df,
    k_values,
):
    results = {}
    top_k_results = {}

    for label in labels:
        n = n_counts[label]
        label_results = {}
        label_results_k = {}

        filtered_embedding_types = [
            "average_cosine_similarity",
            "max_cosine_similarity",
        ]

        for emb in filtered_embedding_types:
            max_df = cosine_max[
                (cosine_max["label"] == label) & (cosine_max["embedding_type"] == emb)
            ]

            mean_df = cosine_mean[
                (cosine_mean["label"] == label) & (cosine_mean["embedding_type"] == emb)
            ]

            # Check for empty DataFrames
            if max_df.empty or mean_df.empty:
                print(
                    f"Warning: No data available for label '{label}' and embedding '{emb}'. Skipping."
                )
                label_results[f"{emb}_max_accuracy"] = np.nan
                label_results[f"{emb}_mean_accuracy"] = np.nan
                for k in k_values:
                    label_results_k[f"{emb}_max_accuracy_top_{k}"] = np.nan
                    label_results_k[f"{emb}_mean_accuracy_top_{k}"] = np.nan
                continue

            # Adjust n to be at most the number of available rows
            n = min(n, len(max_df), len(mean_df))

            # Calculate top_n accuracy
            top_n_max = max_df.nlargest(n, "cosine_similarity")["study_id"]
            top_n_mean = mean_df.nlargest(n, "cosine_similarity")["study_id"]

            label_positives = split_type_labels_df[split_type_labels_df[label] == 1][
                "study_id"
            ]

            # Combine the accuracy metrics with specific names for each embedding type
            label_results[f"{emb}_max_accuracy"] = (
                top_n_max.isin(label_positives).sum() / n if n > 0 else 0
            )

            label_results[f"{emb}_mean_accuracy"] = (
                top_n_mean.isin(label_positives).sum() / n if n > 0 else 0
            )

            # Calculate top_k accuracy for each k in k_values
            for k in k_values:
                k = min(k, len(max_df), len(mean_df))  # Adjust k similarly
                top_k_max = max_df.nlargest(k, "cosine_similarity")["study_id"]
                top_k_mean = mean_df.nlargest(k, "cosine_similarity")["study_id"]

                label_results_k[f"{emb}_max_accuracy_top_{k}"] = (
                    top_k_max.isin(label_positives).sum() / k if k > 0 else 0
                )
                label_results_k[f"{emb}_mean_accuracy_top_{k}"] = (
                    top_k_mean.isin(label_positives).sum() / k if k > 0 else 0
                )

        results[label] = label_results
        top_k_results[label] = label_results_k

    return results, top_k_results


def save_results(results_df, output_path):
    """
    Save results in output df.
    """
    results_df.to_csv(
        output_path, index=False
    )  # Use index=False to avoid duplicate label columns


def calculate_confidence_intervals(data):
    """
    Calculate mean and 95% confidence intervals for resampling data.
    """
    mean = data.mean()
    se = data.std(ddof=1) / np.sqrt(len(data))  # Standard error
    ci_lower = mean - 1.96 * se  # Lower bound of the 95% CI
    ci_upper = mean + 1.96 * se  # Upper bound of the 95% CI
    return mean, ci_lower, ci_upper


def resample_and_calculate_accuracies(
    labels,
    n_counts,
    cosine_max,
    cosine_mean,
    split_type_labels_df,
    k_values,
    num_iterations=1000,
):
    """
    Perform resampling by drawing half of the samples for each label and embedding type, and repeat for `num_iterations`.
    """
    all_results = []

    # Determine the unique study IDs from the combined cosine_max data
    unique_study_ids = cosine_max["study_id"].unique()

    # Ensure there are enough study IDs to sample from
    if len(unique_study_ids) == 0:
        print("Warning: No unique study IDs available for resampling. Exiting.")
        return pd.DataFrame(), pd.DataFrame()

    print(f"Unique study IDs available for resampling: {len(unique_study_ids)}")

    for i in range(num_iterations):
        # Determine the sample size as half of the unique study IDs
        sample_size = max(1, (len(unique_study_ids) // 2))
        print(f"Sample size: {sample_size}")

        # Randomly sample study IDs
        sampled_ids = np.random.choice(
            unique_study_ids, size=sample_size, replace=False
        )

        # Resample the data for the current subset
        sampled_cosine_max = cosine_max[cosine_max["study_id"].isin(sampled_ids)]
        sampled_cosine_mean = cosine_mean[cosine_mean["study_id"].isin(sampled_ids)]
        sampled_split_type_labels_df = split_type_labels_df[
            split_type_labels_df["study_id"].isin(sampled_ids)
        ]

        # Recalculate the number of positive cases in the sampled data
        n_counts_sampled = count_positive_cases(labels, sampled_split_type_labels_df)

        # Ensure that sampled data is not empty
        if sampled_cosine_max.empty or sampled_cosine_mean.empty:
            print(
                f"Warning: No valid data for resampled set in iteration {i}. Skipping this iteration."
            )
            continue

        # Calculate accuracy for the resample
        results, _ = calculate_accuracy_and_top_k(
            labels,
            n_counts_sampled,
            sampled_cosine_max,
            sampled_cosine_mean,
            sampled_split_type_labels_df,
            k_values,
        )

        print(f"Iteration {i} calculated")

        # Store the accuracy results for each embedding type
        for label in results:
            for key, value in results[label].items():
                all_results.append(
                    {
                        "iteration": i,
                        "label": label,
                        "embedding_type": key,
                        "value": value,
                    }
                )

    # Convert all results into a DataFrame
    all_results_df = pd.DataFrame(all_results)

    # Calculate mean and confidence intervals
    mean_ci_results = []

    for label in labels:
        metrics = [
            "average_cosine_similarity_max_accuracy",
            "average_cosine_similarity_mean_accuracy",
            "max_cosine_similarity_max_accuracy",
            "max_cosine_similarity_mean_accuracy",
        ]
        for metric in metrics:
            subset = all_results_df[
                (all_results_df["label"] == label)
                & (all_results_df["embedding_type"] == metric)
            ]

            if not subset.empty:
                subset_values = pd.to_numeric(subset["value"], errors="coerce")
                mean, ci_lower, ci_upper = calculate_confidence_intervals(subset_values)
                mean_ci_results.append(
                    {
                        "label": label,
                        "embedding_type": metric,
                        "mean": mean,
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper,
                    }
                )

    mean_ci_results_df = pd.DataFrame(mean_ci_results)

    return mean_ci_results_df, all_results_df


def main():
    parser = argparse.ArgumentParser(
        description="Calculate accuracy and top-k metrics for cosine similarity data."
    )
    parser.add_argument(
        "-c",
        "--cosine-path",
        type=str,
        required=True,
        help="Path to the cosine similarity CSV file.",
    )
    parser.add_argument(
        "-o",
        "--output-results-path",
        type=str,
        required=True,
        help="Path to save the accuracy results CSV file.",
    )
    parser.add_argument(
        "-t",
        "--output-top-k-results-path",
        type=str,
        required=True,
        help="Path to save the top-k accuracy results CSV file.",
    )
    parser.add_argument(
        "-s",
        "--split-type",
        type=str,
        required=True,
        help="Data split type (e.g., 'validate', 'train').",
    )
    parser.add_argument(
        "-n",
        "--num-iterations",
        type=int,
        default=1000,
        help="Number of resampling iterations. Default is 1000.",
    )
    parser.add_argument(
        "-r",
        "--resampling",
        action="store_true",
        help="Flag to enable resampling for accuracy calculation.",
    )

    args = parser.parse_args()

    # Define labels and embedding types
    labels = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Enlarged Cardiomediastinum",
        "Fracture",
        "Lung Lesion",
        "Lung Opacity",
        "No Finding",
        "Pleural Effusion",
        "Pleural Other",
        "Pneumonia",
        "Pneumothorax",
        "Support Devices",
    ]

    embedding_types = [
        "cosine_similarity_1",
        "cosine_similarity_2",
        "cosine_similarity_3",
        "cosine_similarity_4",
        "average_cosine_similarity",
        "max_cosine_similarity",
    ]

    # Read and process data
    cosine_df, labels_df, split_df = read_data(args.cosine_path)
    split_type_labels_df = filter_split_type_data(split_df, labels_df, args.split_type)
    cosine_df_transformed = transform_cosine_df(cosine_df, labels, embedding_types)
    n_counts = count_positive_cases(labels, split_type_labels_df)
    cosine_max = aggregate_cosine(cosine_df_transformed, "max")
    cosine_mean = aggregate_cosine(cosine_df_transformed, "mean")

    # Define k values for top_k calculations
    k_values = [5, 10, 20]

    if args.resampling:
        # Perform resampling and calculate accuracies
        mean_ci_results_df, all_results_df = resample_and_calculate_accuracies(
            labels,
            n_counts,
            cosine_max,
            cosine_mean,
            split_type_labels_df,
            k_values,
            args.num_iterations,
        )

        # Save mean and CI results
        save_results(
            mean_ci_results_df,
            args.output_results_path.replace(".csv", "_resampling.csv"),
        )

        # Save all resampling results in separate files
        save_results(
            all_results_df,
            args.output_results_path.replace(".csv", "_all_resampling.csv"),
        )

        print("Accuracy with resampling results saved")

    else:
        # Calculate accuracies without resampling
        results, top_k_results = calculate_accuracy_and_top_k(
            labels,
            embedding_types,
            n_counts,
            cosine_max,
            cosine_mean,
            split_type_labels_df,
            k_values,
        )

        # Convert results to DataFrame and save
        results_df = (
            pd.DataFrame(results)
            .T.reset_index()
            .melt(id_vars="index", var_name="metric", value_name="value")
        )
        results_df.columns = ["label", "metric", "value"]
        save_results(results_df, args.output_results_path)

        # Similarly for top_k_results
        top_k_results_df = (
            pd.DataFrame(top_k_results)
            .T.reset_index()
            .melt(id_vars="index", var_name="metric", value_name="value")
        )
        top_k_results_df.columns = ["label", "metric", "value"]
        save_results(top_k_results_df, args.output_top_k_results_path)

        ("Accuracy results saved")


if __name__ == "__main__":
    main()
