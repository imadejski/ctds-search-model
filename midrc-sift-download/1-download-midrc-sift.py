import os
import json
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
from pqdm.processes import pqdm

import gen3
from gen3.auth import Gen3Auth
from gen3.query import Gen3Query
from gen3.tools.download import drs_download

restart_idx = 12497

cred = "/home/songs1/midrc-credentials.json"
api = "https://data.midrc.org"

auth = Gen3Auth(api, refresh_file=cred)
query = Gen3Query(auth)

fpath = "/opt/gpudata/midrc-sift/obj_ids.csv"
if not os.path.exists(fpath):

    def f(l, k):
        return [x[k] for x in l]

    annotations = query.raw_data_download(
        data_type="data_file",
        fields="*",
        filter_object={
            "AND": [
                {"=": {"source_node": "dicom_annotation_file"}},
                {"=": {"annotation_name": "SIFT"}},
                {"=": {"loinc_system": "Chest"}},
            ]
        },
    )

    series = query.raw_data_download(
        data_type="data_file",
        fields="*",
        filter_object={
            "AND": [
                {"IN": {"source_node": ["cr_series_file", "dx_series_file"]}},
                {"IN": {"series_uid": f(annotations, "series_uid")}},
                {"=": {"loinc_system": "Chest"}},
            ]
        },
    )

    a = pd.DataFrame(
        {
            "series_uid": f(annotations, "series_uid"),
            "ann_id": f(annotations, "object_id"),
            "annotation": f(annotations, "file_name"),
        }
    )
    i = pd.DataFrame(
        {
            "series_uid": f(series, "series_uid"),
            "im_id": f(series, "object_id"),
            "image": f(series, "file_name"),
        }
    )
    df = a.merge(i, on="series_uid", how="inner").drop_duplicates(
        "series_uid", keep=False
    )

    df.to_csv(fpath, index=False)
else:
    df = pd.read_csv(fpath)

output_dir = "/opt/gpudata/midrc-sift"
epath = os.path.join(output_dir, "failed.csv")
if not os.path.exists(epath):
    with open(epath, "w") as f:
        f.write("obj_id\n")
obj_ids = pd.concat(
    [
        df[["series_uid", "ann_id"]].rename(columns={"ann_id": "obj_id"}),
        df[["series_uid", "im_id"]].rename(columns={"im_id": "obj_id"}),
    ]
)

i = restart_idx
attempts = 0
max_attempts = 3

with tqdm(total=len(obj_ids)) as pbar:
    pbar.update(i)
    while i < len(obj_ids):
        row = obj_ids.iloc[i]
        series_uid = row["series_uid"]
        obj_id = row["obj_id"]

        download_list = [drs_download.Downloadable(obj_id)]
        downloader = drs_download.DownloadManager(
            hostname="data.midrc.org",
            auth=auth,
            download_list=download_list,
        )
        res = downloader.download(
            object_list=download_list,
            save_directory=os.path.join(output_dir, series_uid),
        )
        attempts += 1

        if res[obj_id].status == "downloaded":
            i += 1
            attempts = 0
            pbar.update()
        elif attempts == max_attempts:
            i += 1
            attempts = 0
            pbar.update()
            with open(epath, "a") as f:
                f.write(f"{obj_id}\n")
        # else retry same idx
