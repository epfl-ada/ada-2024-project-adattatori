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

def read_tsv_from_zip_comp(zip_path, file_name_in_zip):
    # Unzips the file and reads the TSV
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        with zip_ref.open(file_name_in_zip) as tsv_file:
            # Read the file directly as a TSV
            df = bdb = pd.read_csv(zip_path, sep = '\t', on_bad_lines = 'skip', low_memory = False, dtype={'BindingDB Target Chain Sequence.1': 'object',
       'DrugBank ID of Ligand': 'object',
       'IC50 (nM)': 'object',
       'KEGG ID of Ligand': 'object',
       'Kd (nM)': 'object',
       'Ki (nM)': 'object',
       'Ligand HET ID in PDB': 'object',
       'PDB ID(s) for Ligand-Target Complex': 'object',
       'PDB ID(s) of Target Chain.1': 'object',
       'PMID': 'float64',
       'Patent Number': 'object',
       'UniProt (SwissProt) Entry Name of Target Chain.1': 'object',
       'UniProt (SwissProt) Primary ID of Target Chain.1': 'object',
       'UniProt (SwissProt) Recommended Name of Target Chain.1': 'object',
       'UniProt (SwissProt) Secondary ID(s) of Target Chain': 'object',
       'UniProt (SwissProt) Secondary ID(s) of Target Chain.1': 'object',
       'UniProt (TrEMBL) Entry Name of Target Chain': 'object',
       'UniProt (TrEMBL) Primary ID of Target Chain': 'object',
       'UniProt (TrEMBL) Secondary ID(s) of Target Chain': 'object',
       'UniProt (TrEMBL) Submitted Name of Target Chain': 'object'})
    return df


def get_dataset2():
    root = find_repo_root()
    path_to_data = os.path.join(root,'data','BindingDB_All.zip')
    filename = 'BindingDB_All.tsv'
    df = read_tsv_from_zip_comp(path_to_data, filename)
    return df
