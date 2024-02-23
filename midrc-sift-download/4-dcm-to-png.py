import os
import numpy as np
import pandas as pd
import pydicom
from pqdm.processes import pqdm
from PIL import Image


data_dir = "/opt/gpudata/midrc-sift"
dcm_dir = os.path.join(data_dir, "dcm")
png_dir = os.path.join(data_dir, "png")
series_uids = sorted(os.listdir(dcm_dir))
obj_ids = pd.read_csv(os.path.join(data_dir, "obj_ids.csv"))
dcm_csv = os.path.join(data_dir, "annotated_dcms.csv")

annotated_dcms = pd.read_csv(dcm_csv)

samples = annotated_dcms.to_numpy()


def convert(args):
    series_uid, image_uid, dcm_path = args
    series_dir = os.path.join(png_dir, series_uid)
    os.makedirs(series_dir)

    dcm = pydicom.dcmread(dcm_path)
    im_arr = dcm.pixel_array.astype(float)

    # check not multi image
    if im_arr.ndim > 2:
        assert im_arr.shape[2] == 3  # if not 2D (grayscale), must be rgb

    # convert to uint8
    im_scaled = (np.maximum(im_arr, 0) / im_arr.max()) * 255.0
    im_scaled = np.uint8(im_scaled)
    im = Image.fromarray(im_scaled)

    im.save(os.path.join(series_dir, f"{image_uid}.png"))


pqdm(samples, convert, n_jobs=20)
