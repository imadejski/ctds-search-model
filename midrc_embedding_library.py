import os
from enum import Enum, unique
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests
import torch
from health_multimodal.image import ImageInferenceEngine
from health_multimodal.image.data.transforms import (
    create_chest_xray_transform_for_inference,
)
from health_multimodal.image.model.pretrained import get_biovil_t_image_encoder
from health_multimodal.text.utils import BertEncoderType, get_bert_inference
from health_multimodal.vlp.inference_engine import ImageTextInferenceEngine
from torchvision.datasets.utils import check_integrity
from tqdm import tqdm

RESIZE = 512
CENTER_CROP_SIZE = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def create_paths_df(directory_path):
    """Creates the dataframe with file paths of each image"""
    data = []

    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".png"):
                folder_name = os.path.basename(root)
                file_path = os.path.join(root, file)
                data.append(
                    {
                        "folder_name": folder_name,
                        "image_name": file,
                        "file_path": file_path,
                    }
                )

    df = pd.DataFrame(data, columns=["folder_name", "image_name", "file_path"])

    return df


def _get_image_inference_engine():
    """
    Defines image inference model from BioVIL-T image encoder.
    Applies resizing and cropping to image.
    """
    image_inference = ImageInferenceEngine(
        image_model=get_biovil_t_image_encoder().to(device),
        transform=create_chest_xray_transform_for_inference(
            resize=RESIZE, center_crop_size=CENTER_CROP_SIZE
        ),
    )
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

    image_embedding = inference_engine.get_projected_global_embedding(
        image_path=Path(image_path)
    )
    np_img_embedding = convert_tensor_to_np_array(image_embedding)
    return np_img_embedding


def main():
    image_inference = _get_image_inference_engine()

    img_folder_path = "/opt/gpudata/midrc-sift/png"

    embedding_library_output_path = (
        "/home/imadejski/ctds-search-model/midrc_embedding_library.csv"
    )

    path_df = create_paths_df(img_folder_path)

    path_df["embedding"] = np.nan

    tqdm.pandas(desc="Calculating Embeddings")
    path_df["embedding"] = path_df["file_path"].progress_apply(
        get_image_embedding, inference_engine=image_inference
    )

    path_df.to_csv(embedding_library_output_path, index=False)

    return None


if __name__ == "__main__":
    main()
