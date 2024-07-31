from math import ceil, floor
from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
from health_multimodal.image import ImageInferenceEngine
from health_multimodal.image.data.transforms import (
    create_chest_xray_transform_for_inference,
)
from health_multimodal.image.model.pretrained import get_biovil_image_encoder
from health_multimodal.text import TextInferenceEngine
from health_multimodal.text.utils import BertEncoderType, get_bert_inference
from health_multimodal.vlp.inference_engine import ImageTextInferenceEngine
from scipy import ndimage
from torchvision.datasets.utils import check_integrity
from transformers import BertForMaskedLM, BertTokenizer

RESIZE = 512
CENTER_CROP_SIZE = 512


def _get_vlp_inference_engine() -> ImageTextInferenceEngine:
    """
    Creates an returns an instance of the ImageTextInferenceEngine
    """
    image_inference = ImageInferenceEngine(
        image_model=get_biovil_image_encoder(),
        transform=create_chest_xray_transform_for_inference(
            resize=RESIZE, center_crop_size=CENTER_CROP_SIZE
        ),
    )
    img_txt_inference = ImageTextInferenceEngine(
        image_inference_engine=image_inference,
        text_inference_engine=get_bert_inference(BertEncoderType.CXR_BERT),
    )
    return img_txt_inference


def np_array_to_torch_tensor(np_array):
    """
    Takes a nump array and converts it to a torch tensor with datatype float32
    """
    torch_tensor = torch.tensor(np_array, dtype=torch.float32)
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


def average_embedding(search_queries, inference_engine):
    """
    Takes a list of search queries and returns the average embedding
    """
    avg_embedding = inference_engine.get_embeddings_from_prompt(search_queries)

    return avg_embedding


def create_search_queries_embedding(search_queries, inference_engine):
    """
    Takes a list of search queries and returns list of embeddings for each search query
    """
    search_queries_embeddings = []
    for query in search_queries:
        embedding = inference_engine.get_embeddings_from_prompt(query)
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
    text_inference = get_bert_inference(BertEncoderType.CXR_BERT)

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

    mimic_embedding_library_path = "/home/imadejski/ctds-search-model/data/mimic/embedding_libraries/mimic_validate_embedding_library_biovil.csv"
    mimic_embeddings_pd = img_embeddings_df(mimic_embedding_library_path)

    for label in labels:
        pos_search_queries = create_pos_search_queries_for_label(label)
        pos_embeddings = create_search_queries_embedding(
            pos_search_queries, text_inference
        )
        avg_pos_embedding = average_embedding(pos_search_queries, text_inference)

        for i in range(len(pos_search_queries)):
            mimic_embeddings_pd[f"{label} cosine_similarity_{i+1}"] = np.nan

        mimic_embeddings_pd[f"{label} average_cosine_similarity"] = np.nan
        mimic_embeddings_pd[f"{label} max_cosine_similarity"] = np.nan

        for index, row in mimic_embeddings_pd.iterrows():
            img_embedding_string = row["embedding"]
            img_embedding_np = convert_string_to_np(img_embedding_string)
            img_embedding = np_array_to_torch_tensor(img_embedding_np)

            cosine_similarities = []
            for i, pos_embedding in enumerate(pos_embeddings):
                cosine_similarity = find_cosine_similarity(img_embedding, pos_embedding)
                mimic_embeddings_pd.at[
                    index, f"{label} cosine_similarity_{i+1}"
                ] = cosine_similarity
                cosine_similarities.append(cosine_similarity)

            if cosine_similarities:
                average_of_cosine_similarities = np.mean(cosine_similarities)
                mimic_embeddings_pd.at[
                    index, f"{label} average_cosine_similarity"
                ] = average_of_cosine_similarities

                max_of_cosine_similarities = np.max(cosine_similarities)
                mimic_embeddings_pd.at[
                    index, f"{label} max_cosine_similarity"
                ] = max_of_cosine_similarities

    output_file_path = "/home/imadejski/ctds-search-model/data/mimic/cosine_similarity/mimic_validate_multiple_query_single_pos_label_cosine_similarity_biovil.csv"
    mimic_embeddings_pd.to_csv(output_file_path, index=False)


if __name__ == "__main__":
    main()
