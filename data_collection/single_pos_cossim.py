import tempfile
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from health_multimodal.common.visualization import plot_phrase_grounding_similarity_map
from health_multimodal.image import get_image_inference
from health_multimodal.image.utils import ImageModelType
from health_multimodal.text import get_bert_inference
from health_multimodal.text.utils import BertEncoderType
from health_multimodal.vlp import ImageTextInferenceEngine
from IPython.display import Markdown, display

text_inference = get_bert_inference(BertEncoderType.BIOVIL_T_BERT)
image_inference = get_image_inference(ImageModelType.BIOVIL_T)

image_text_inference = ImageTextInferenceEngine(
    image_inference_engine=image_inference,
    text_inference_engine=text_inference,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_text_inference.to(device)


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


def find_cossim(image_path_name, text_prompt):
    """Finds cosine similarity between an image and text given the image path and text prompt,
    computes embeddings for each and calculates cosine similarity"""
    image_path = Path(image_path_name)
    similarity_score = image_text_inference.get_similarity_score_from_raw_data(
        image_path=image_path,
        query_text=text_prompt,
    )
    return similarity_score


def cossim_batch_for_label(df, label):
    for index, row in df.iterrows():
        image_path = row["file_path"]
        text_prompt = str(label + " seen")
        cossim = find_cossim(image_path, text_prompt)
        df.loc[index, label] = cossim

    return None


def main():
    chexpert_labels = [
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

    split_file_path = "/opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-split.csv.gz"

    validate_df = create_validate_df(split_file_path)
    validate_df = create_paths(validate_df)

    for label in chexpert_labels:
        validate_df[label] = None
        cossim_batch_for_label(validate_df, label)

    avg_df = validate_df.groupby("study_id")[chexpert_labels].mean().reset_index()

    dicom_output_csv_path = (
        "/home/imadejski/ctds-search-model/dicom_cosine_similarity_single_pos_label.csv"
    )
    avg_output_csv_path = (
        "/home/imadejski/ctds-search-model/avg_cosine_similarity_single_pos_label.csv"
    )

    validate_df.to_csv(dicom_output_csv_path, index=False)
    avg_df.to_csv(avg_output_csv_path, index=False)

    return None


if __name__ == "__main__":
    main()
