from enum import Enum, unique
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests
import torch
from medcap.models import (
    ImageTextMultiScaleContraster,
    ImageTextMultiScaleContrasterConfig,
    InferenceEngine,
)
from torchvision.datasets.utils import check_integrity
from tqdm import tqdm

RESIZE = 512
CENTER_CROP_SIZE = 512

base_model_path = "microsoft/BiomedVLP-CXR-BERT-specialized"
train_full_run_model_path = "/opt/gpudata/imadejski/image_text_global_local_alignment/model_070924_train_full_run"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def create_validate_df(file_path):
    """Creates the dataframe with the validation data and initializes a file_path column"""
    split_df = pd.read_csv(file_path)
    validate_df = split_df[split_df["split"].isin(["validate"])]
    return validate_df


def create_paths(df):
    """Creates a column with the path for each individual image,
    takes a datafram with columns subject_ids, study_ids, and dicom_ids"""
    for index, row in df.iterrows():
        # retrieve each sub folder value to get path
        patient_id = row["subject_id"]
        study_id = row["study_id"]
        dicom_id = row["dicom_id"]

        # assign path in df
        path = (
            "/opt/gpudata/mimic-cxr/jpg/p"
            + str(patient_id)[:2]
            + "/p"
            + str(patient_id)
            + "/s"
            + str(study_id)
            + "/"
            + str(dicom_id)
            + ".jpg"
        )
        df.loc[index, "file_path"] = path
        df.loc[index, "study_id"] = study_id
    return df


def _get_image_inference_engine():
    """
    Defines image inference model from BioVIL-T image encoder.
    Applies resizing and cropping to image.
    """
    image_inference = InferenceEngine(train_full_run_model_path, base_model_path)
    return image_inference


def convert_tensor_to_np_array(tensor):
    """
    Returns numpy array from Torch tensor.
    """
    if tensor.is_cuda:
        tensor = tensor.cpu()

    numpy_array = tensor.numpy()
    return numpy_array


def get_image_embedding(image_path, inference_engine):
    """
    Returns numpy array of l2-normalized global image embedding from
    image inference model
    """

    image_embedding = inference_engine.get_projected_global_embeddings(image_path)
    np_img_embedding = convert_tensor_to_np_array(image_embedding)
    return np_img_embedding


def main():
    image_inference = _get_image_inference_engine()

    split_file_path = "/opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-split.csv.gz"

    embedding_library_output_path = "/opt/gpudata/imadejski/search-model/ctds-search-model/mimic_validate_embedding_library_070924_model.csv"

    validate_df = create_validate_df(split_file_path)
    validate_df = create_paths(validate_df)

    validate_df["embedding"] = np.nan

    tqdm.pandas(desc="Calculating Embeddings")
    validate_df["embedding"] = validate_df["file_path"].progress_apply(
        get_image_embedding, inference_engine=image_inference
    )

    validate_df.to_csv(embedding_library_output_path, index=False)

    return None


if __name__ == "__main__":
    main()
