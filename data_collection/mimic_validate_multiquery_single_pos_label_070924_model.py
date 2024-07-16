from math import ceil, floor
from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
from medcap.models import (
    ImageTextMultiScaleContraster,
    ImageTextMultiScaleContrasterConfig,
    InferenceEngine,
)
from scipy import ndimage
from torchvision.datasets.utils import check_integrity
from transformers import BertForMaskedLM, BertTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_inference_engine(model_checkpoint_path, tokenizer_path) -> InferenceEngine:
    """
    Creates an returns an instance of the InferenceEngine
    """
    inference_engine = InferenceEngine(model_checkpoint_path, tokenizer_path)
    return inference_engine


def np_array_to_torch_tensor(np_array, device):
    """
    Takes a nump array and converts it to a torch tensor with datatype float32
    """
    torch_tensor = torch.tensor(np_array, dtype=torch.float32, device=device)
    return torch_tensor


def img_embeddings_df(path):
    """
    Takes a path for a file path for a csv that holds information about image imbeddings
    Returns a Pandas dataframe
    """
    embeddings_pd = pd.read_csv(path)
    return embeddings_pd


def create_pos_search_queries_for_label(label):
    """
    Takes a label (string) and creates a list of positive search queries for a given
    """
    pos_search_queries = [
        f"Findings consistent with {label}",
        f"Findings suggesting {label}",
        f"Findings are most compatible with {label}",
        f"{label} seen",
    ]
    return pos_search_queries


def create_search_queries_embedding(search_queries, inference_engine, device=device):
    """
    Takes a list of search queries and returns list of embeddings for each search query
    """
    search_queries_embeddings = []
    for query in search_queries:
        embedding = inference_engine.get_projected_text_embedding(query).to(device)
        search_queries_embeddings.append(embedding)
    return search_queries_embeddings


def find_cosine_similarity(img_embedding, search_query_embedding):
    """
    Takes an image embedding and search query embedding as torch tensors and returns the cosine
    similarity score
    Img embedding size should be ([128]) and text embedding is ([1, 128])
    """
    img_embedding_reshaped = img_embedding.reshape(1, 128)

    text_embedding = search_query_embedding.mean(dim=0)
    text_embedding = F.normalize(text_embedding, dim=0, p=2)

    cos_similarity = img_embedding @ text_embedding.t()
    return cos_similarity.item()


def convert_string_to_np(embedding_str):
    """
    Converts a string to a numpy array
    """
    return np.fromstring(embedding_str[1:-1], sep=" ")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_model_path = "microsoft/BiomedVLP-CXR-BERT-specialized"
    train_full_run_model_path = "/opt/gpudata/imadejski/image_text_global_local_alignment/model_070924_train_full_run"

    checkpoint_inference_engine = _get_inference_engine(
        train_full_run_model_path, base_model_path
    )

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

    mimic_validate_embedding_library_070924_model_path = "/opt/gpudata/imadejski/search-model/ctds-search-model/data/mimic_validate_embedding_library_070924_model.csv"
    mimic_validate_embeddings_pd = img_embeddings_df(
        mimic_validate_embedding_library_070924_model_path
    )

    for label in labels:
        pos_search_queries = create_pos_search_queries_for_label(label)
        pos_embeddings = create_search_queries_embedding(
            pos_search_queries, checkpoint_inference_engine, device
        )

        for i in range(len(pos_search_queries)):
            mimic_validate_embeddings_pd[f"{label} cosine_similarity_{i+1}"] = np.nan

        mimic_validate_embeddings_pd[f"{label} average_cosine_similarity"] = np.nan
        mimic_validate_embeddings_pd[f"{label} max_cosine_similarity"] = np.nan

        for index, row in mimic_validate_embeddings_pd.iterrows():
            img_embedding_string = row["embedding"]
            img_embedding_np = convert_string_to_np(img_embedding_string)
            img_embedding = np_array_to_torch_tensor(img_embedding_np, device)

            cosine_similarities = []
            for i, pos_embedding in enumerate(pos_embeddings):
                cosine_similarity = find_cosine_similarity(img_embedding, pos_embedding)
                mimic_validate_embeddings_pd.at[
                    index, f"{label} cosine_similarity_{i+1}"
                ] = cosine_similarity
                cosine_similarities.append(cosine_similarity)

            if cosine_similarities:
                average_of_cosine_similarities = np.mean(cosine_similarities)
                mimic_validate_embeddings_pd.at[
                    index, f"{label} average_cosine_similarity"
                ] = average_of_cosine_similarities

                max_of_cosine_similarities = np.max(cosine_similarities)
                mimic_validate_embeddings_pd.at[
                    index, f"{label} max_cosine_similarity"
                ] = max_of_cosine_similarities

    output_file_path = "/opt/gpudata/imadejski/search-model/ctds-search-model/data/mimic_validate_multiquery_single_pos_label_070924_cosine_similarity.csv"
    mimic_validate_embeddings_pd.to_csv(output_file_path, index=False)


if __name__ == "__main__":
    main()
