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

RESIZE = 512
CENTER_CROP_SIZE = 512


def create_validate_df(file_path):
    """Creates the dataframe with the validation data and initializes a file_path column"""
    split_df = pd.read_csv(file_path)
    validate_df = split_df[split_df["split"] == "validate"]
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
    image_inference = ImageInferenceEngine(
        image_model=get_biovil_t_image_encoder(),
        transform=create_chest_xray_transform_for_inference(
            resize=RESIZE, center_crop_size=CENTER_CROP_SIZE
        ),
    )
    return image_inference


def get_image_embedding(image_path):
    image_inference = _get_image_inference_engine()
    image_embedding = image_inference.get_projected_global_embedding(
        image_path=Path(image_path)
    )
    np_img_embedding = convert_tensor_to_np_array(image_embedding)
    return np_img_embedding


def convert_tensor_to_np_array(tensor):
    if tensor.is_cuda:
        tensor = tensor.cpu()

    numpy_array = tensor.numpy()
    return numpy_array


def main():
    split_file_path = "/opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-split.csv.gz"

    embedding_library_output_path = (
        "/home/imadejski/ctds-search-model/mimic_validate_embedding_library.csv"
    )

    validate_df = create_validate_df(split_file_path)
    validate_df = create_paths(validate_df)

    validate_df["embedding"] = np.nan

    validate_df["embedding"] = validate_df["file_path"].apply(get_image_embedding)

    validate_df.to_csv(embedding_library_output_path, index=False)

    return None


if __name__ == "__main__":
    main()
