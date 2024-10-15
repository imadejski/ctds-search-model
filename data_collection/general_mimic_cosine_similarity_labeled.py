import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from medcap.models import InferenceEngine
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_inference_engine(model_checkpoint_path, tokenizer_path) -> InferenceEngine:
    """
    Creates and returns an instance of the InferenceEngine
    """
    inference_engine = InferenceEngine(model_checkpoint_path, tokenizer_path)
    return inference_engine


def np_array_to_torch_tensor(np_array, device):
    """
    Takes a numpy array and converts it to a torch tensor with datatype float32
    """
    torch_tensor = torch.tensor(np_array, dtype=torch.float32, device=device)
    return torch_tensor


def img_embeddings_df(path, split_type):
    """
    Takes a file path for a CSV that holds information about image embeddings
    Filters the dataframe based on the split type
    Returns a Pandas dataframe
    """
    embeddings_pd = pd.read_csv(path)
    embeddings_pd = embeddings_pd[embeddings_pd["split"] == split_type]
    return embeddings_pd


def create_pos_search_queries_for_label(label):
    """
    Takes a label (string) and creates a list of positive search queries for the given label
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
    Takes a list of search queries and returns a list of embeddings for each search query
    """
    search_queries_embeddings = []
    for query in search_queries:
        embedding = inference_engine.get_projected_text_embedding(query).to(device)
        search_queries_embeddings.append(embedding)
    return search_queries_embeddings


def find_cosine_similarity(img_embedding, search_query_embedding):
    """
    Takes an image embedding and a search query embedding as torch tensors and returns the cosine
    similarity score
    Image embedding size should be ([128]) and text embedding is ([1, 128])
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


def main(model_checkpoint_path, embedding_library_path, output_file_path, split_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_model_path = "microsoft/BiomedVLP-CXR-BERT-specialized"

    checkpoint_inference_engine = _get_inference_engine(
        model_checkpoint_path, base_model_path
    )

    labels = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Enlarged Cardiomediastinum",
        "Fracture",
        "Lung Lesion",
        "Airspace Opacity",
        "No Finding",
        "Pleural Effusion",
        "Pleural Other",
        "Pneumonia",
        "Pneumothorax",
        "Support Devices",
    ]

    embeddings_library_pd = img_embeddings_df(embedding_library_path, split_type)

    for label in labels:
        pos_search_queries = create_pos_search_queries_for_label(label)
        pos_embeddings = create_search_queries_embedding(
            pos_search_queries, checkpoint_inference_engine, device
        )

        for i in range(len(pos_search_queries)):
            embeddings_library_pd[f"{label} cosine_similarity_{i+1}"] = np.nan

        embeddings_library_pd[f"{label} average_cosine_similarity"] = np.nan
        embeddings_library_pd[f"{label} max_cosine_similarity"] = np.nan

        for index, row in embeddings_library_pd.iterrows():
            img_embedding_string = row["embedding"]
            img_embedding_np = convert_string_to_np(img_embedding_string)
            img_embedding = np_array_to_torch_tensor(img_embedding_np, device)

            cosine_similarities = []
            for i, pos_embedding in enumerate(pos_embeddings):
                cosine_similarity = find_cosine_similarity(img_embedding, pos_embedding)
                embeddings_library_pd.at[
                    index, f"{label} cosine_similarity_{i+1}"
                ] = cosine_similarity
                cosine_similarities.append(cosine_similarity)

            if cosine_similarities:
                average_of_cosine_similarities = np.mean(cosine_similarities)
                embeddings_library_pd.at[
                    index, f"{label} average_cosine_similarity"
                ] = average_of_cosine_similarities

                max_of_cosine_similarities = np.max(cosine_similarities)
                embeddings_library_pd.at[
                    index, f"{label} max_cosine_similarity"
                ] = max_of_cosine_similarities

    embeddings_library_pd.to_csv(output_file_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get paths")
    parser.add_argument(
        "model_checkpoint_path",
        type=str,
        help="Path to the fine-tuned model checkpoint",
    )
    parser.add_argument(
        "embedding_library_path",
        type=str,
        help="Path to the embedding library CSV file",
    )
    parser.add_argument(
        "output_file_path", type=str, help="Path to save the output CSV file"
    )
    parser.add_argument(
        "split_type", type=str, help="Type of data split (e.g., validate, train, test)"
    )

    args = parser.parse_args()
    main(
        args.model_checkpoint_path,
        args.embedding_library_path,
        args.output_file_path,
        args.split_type,
    )
