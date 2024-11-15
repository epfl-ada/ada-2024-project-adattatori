import numpy as np
import os
import pandas as pd
import zipfile


def find_repo_root():
    currpath = os.path.dirname(os.path.abspath(__file__))
    while currpath != os.path.dirname(currpath):  # As long as we dont reach the right root
        if os.path.exists(os.path.join(currpath, '.git')):
            return currpath
        currpath = os.path.dirname(currpath)
    raise Exception("repo root not found")


def read_tsv_from_zip(zip_path, file_name_in_zip):
    # Unzips the file and reads the TSV
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        with zip_ref.open(file_name_in_zip) as tsv_file:
            # Read the file directly as a TSV
            df = pd.read_csv(tsv_file, sep = '\t', low_memory = False)
    return df

def get_dataset():
    root = find_repo_root()
    path_to_data = os.path.join(root,'data','BindingSTD.zip')
    filename = 'BindingSTD.tsv'
    df = read_tsv_from_zip(path_to_data, filename)
    return df
