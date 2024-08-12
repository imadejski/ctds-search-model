import argparse
from typing import Literal

import pandas as pd


def read_data(cosine_path):
    """
    Reads in csvs with cosine similarity scores, ground-truth labels, and split
    """
    cosine_df = pd.read_csv(cosine_path)
    labels_df = pd.read_csv("/opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv.gz")
    split_df = pd.read_csv("/opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-split.csv.gz")
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
    return pd.concat(data, ignore_index=True)


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
        return (
            df.groupby(["study_id", "label", "embedding_type"])["cosine_similarity"]
            .max()
            .reset_index()
        )
    elif method == "mean":
        return (
            df.groupby(["study_id", "label", "embedding_type"])["cosine_similarity"]
            .mean()
            .reset_index()
        )
    else:
        raise ValueError("Aggregation method must be 'max' or 'mean'")


def calculate_accuracy_and_top_k(
    labels,
    embedding_types,
    n_counts,
    cosine_max,
    cosine_mean,
    split_type_labels_df,
    k_values,
):
    """
    Calculates the top n and topk accuracy values
    """
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

            # Calculate top_n accuracy
            top_n_max = max_df.nlargest(n, "cosine_similarity")["study_id"]
            top_n_mean = mean_df.nlargest(n, "cosine_similarity")["study_id"]

            label_positives = split_type_labels_df[split_type_labels_df[label] == 1][
                "study_id"
            ]

            label_results[f"{emb}_max_accuracy"] = (
                top_n_max.isin(label_positives).sum() / n
            )
            label_results[f"{emb}_mean_accuracy"] = (
                top_n_mean.isin(label_positives).sum() / n
            )

            # Calculate top_k accuracy for each k in k_values
            for k in k_values:
                top_k_max = max_df.nlargest(k, "cosine_similarity")["study_id"]
                top_k_mean = mean_df.nlargest(k, "cosine_similarity")["study_id"]

                label_results_k[f"{emb}_max_accuracy_top_{k}"] = (
                    top_k_max.isin(label_positives).sum() / k
                )
                label_results_k[f"{emb}_mean_accuracy_top_{k}"] = (
                    top_k_mean.isin(label_positives).sum() / k
                )

        results[label] = label_results
        top_k_results[label] = label_results_k

    return results, top_k_results


def save_results(results, output_path):
    """
    Save results in output df
    """
    results_df = pd.DataFrame.from_dict(results, orient="index")
    results_df.to_csv(output_path, index=True, index_label="Label")


def main():
    parser = argparse.ArgumentParser(description="Process some paths.")
    parser.add_argument(
        "cosine_path", type=str, help="Path to the cosine similarity CSV file"
    )
    parser.add_argument(
        "output_results_path",
        type=str,
        help="Path to save the accuracy results CSV file",
    )
    parser.add_argument(
        "output_top_k_results_path",
        type=str,
        help="Path to save the top-k accuracy results CSV file",
    )
    parser.add_argument("split_type", type=str, help="Data split (e.g., validate)")

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

    # Calculate accuracies and top_k accuracies
    results, top_k_results = calculate_accuracy_and_top_k(
        labels,
        embedding_types,
        n_counts,
        cosine_max,
        cosine_mean,
        split_type_labels_df,
        k_values,
    )

    save_results(results, args.output_results_path)
    save_results(top_k_results, args.output_top_k_results_path)


if __name__ == "__main__":
    main()
