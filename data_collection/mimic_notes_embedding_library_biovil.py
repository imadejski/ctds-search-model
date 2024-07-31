import numpy as np
import pandas as pd
import torch
from health_multimodal.image import ImageInferenceEngine
from health_multimodal.image.data.transforms import (
    create_chest_xray_transform_for_inference,
)
from health_multimodal.image.model.pretrained import get_biovil_t_image_encoder
from health_multimodal.text.utils import BertEncoderType, get_bert_inference
from health_multimodal.vlp.inference_engine import ImageTextInferenceEngine
from tqdm import tqdm

RESIZE = 512
CENTER_CROP_SIZE = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def _get_vlp_inference_engine() -> ImageTextInferenceEngine:
    """
    Creates and returns an instance of the ImageTextInferenceEngine.
    """
    image_inference = ImageInferenceEngine(
        image_model=get_biovil_t_image_encoder(),
        transform=create_chest_xray_transform_for_inference(
            resize=RESIZE, center_crop_size=CENTER_CROP_SIZE
        ),
    )
    img_txt_inference = ImageTextInferenceEngine(
        image_inference_engine=image_inference,
        text_inference_engine=get_bert_inference(BertEncoderType.BIOVIL_T_BERT),
    )
    return img_txt_inference


def create_paths(df):
    """Creates a column with the path for each individual image,
    takes a dataframe with columns subject_ids and study_ids."""
    for index, row in df.iterrows():
        # retrieve each sub folder value to get path
        patient_id = row["subject_id"]
        study_id = row["study_id"]

        # assign path in df
        path = (
            "/opt/gpudata/mimic-cxr/notes/p"
            + str(patient_id)[:2]
            + "/p"
            + str(patient_id)
            + "/s"
            + str(study_id)
            + ".txt"
        )
        df.loc[index, "file_path"] = path
        df.loc[index, "study_id"] = study_id
    return df


def convert_tensor_to_np_array(tensor):
    """
    Returns numpy array from Torch tensor.
    """
    if tensor.is_cuda:
        tensor = tensor.cpu()

    numpy_array = tensor.detach().numpy()
    return numpy_array


def get_text_embedding(text_path, inference_engine):
    """
    Returns numpy array of l2-normalized global image embedding from
    image inference model.
    """
    with open(text_path, "r") as file:
        text_content = file.read()

    image_embedding = inference_engine.get_embeddings_from_prompt(text_content)
    np_img_embedding = convert_tensor_to_np_array(image_embedding)
    return np_img_embedding


def main():
    output_path = "/opt/gpudata/imadejski/search-model/ctds-search-model/data/mimic_notes_embedding_library_biovil"
    image_inference = _get_vlp_inference_engine()

    split_file_path = "/opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-split.csv.gz"
    split_df = pd.read_csv(split_file_path)
    split_df = create_paths(split_df)

    split_df["note_embedding"] = np.nan

    tqdm.pandas(desc="Calculating Embeddings")
    split_df["note_embedding"] = split_df["file_path"].progress_apply(
        get_text_embedding, inference_engine=image_inference
    )

    split_df.to_csv(output_path, index=False)

    return None


if __name__ == "__main__":
    main()
