from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem import rdMolDescriptors
import numpy as np
from rdkit import DataStructs
import pandas as pd
from scipy.spatial.distance import dice

def smiles_to_mol(smiles):
    """Convert SMILES string to RDKit Mol object."""
    try:
        return Chem.MolFromSmiles(smiles)
    except:
        return None

"""
def calculate_descriptors(mol):

    """"""Calculate RDKit molecular descriptors for a given molecule.""""""
    if mol is None:
        #return pd.Series([None] * 5, index=["MolWt", "LogP", "NumHDonors", "NumHAcceptors", "TPSA"])
        return pd.Series([None] * 12, index=[
            "MolWt", "LogP", "NumHDonors", "NumHAcceptors", "TPSA", 
            "NumRotatableBonds", "NumAromaticRings", "NumAliphaticRings", 
            "FractionCSP3", "RingCount", "FormalCharge", "NumRadicalElectrons"])
    return pd.Series({
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "NumHDonors": Descriptors.NumHDonors(mol),
        "NumHAcceptors": Descriptors.NumHAcceptors(mol),
        "TPSA": Descriptors.TPSA(mol),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
        "NumAromaticRings": Descriptors.NumAromaticRings(mol),
        "NumAliphaticRings": Descriptors.NumAliphaticRings(mol),
        "FractionCSP3": Descriptors.FractionCSP3(mol),
        "RingCount": Descriptors.RingCount(mol),
        "FormalCharge": Chem.GetFormalCharge(mol),
        "NumRadicalElectrons": Descriptors.NumRadicalElectrons(mol)
    })
"""

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdMolDescriptors import CalcNumAtomStereoCenters, CalcNumUnspecifiedAtomStereoCenters
from rdkit.Chem.GraphDescriptors import BalabanJ, BertzCT
from rdkit.Chem.Crippen import MolMR
from rdkit.Chem.Lipinski import NumSaturatedRings, NumSaturatedCarbocycles, NumSaturatedHeterocycles
from rdkit.Chem.rdMolDescriptors import CalcNumSpiroAtoms, CalcNumBridgeheadAtoms, CalcChi0n, CalcChi0v, CalcChi1n, CalcChi1v

def calculate_descriptors(mol):
    """Calculate RDKit molecular descriptors for a given molecule."""
    if mol is None:
        # Define all descriptor names in the index
        return pd.Series([None] * 30, index=[
            "MolWt", "LogP", "NumHDonors", "NumHAcceptors", "TPSA", 
            "NumRotatableBonds", "NumAromaticRings", "NumAliphaticRings", 
            "FractionCSP3", "RingCount", "FormalCharge", "NumRadicalElectrons",
            "BalabanJ", "BertzCT", "MolMR", 
            "NumAtomStereoCenters", "NumUnspecifiedAtomStereoCenters", 
            "NumSpiroAtoms", "NumBridgeheadAtoms", 
            "Chi0n", "Chi0v", "Chi1n", "Chi1v",
            "NumSaturatedRings", "NumSaturatedCarbocycles", "NumSaturatedHeterocycles"
        ])

    return pd.Series({
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "NumHDonors": Descriptors.NumHDonors(mol),
        "NumHAcceptors": Descriptors.NumHAcceptors(mol),
        "TPSA": Descriptors.TPSA(mol),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
        "NumAromaticRings": Descriptors.NumAromaticRings(mol),
        "NumAliphaticRings": Descriptors.NumAliphaticRings(mol),
        "FractionCSP3": Descriptors.FractionCSP3(mol),
        "RingCount": Descriptors.RingCount(mol),
        "FormalCharge": Chem.GetFormalCharge(mol),
        "NumRadicalElectrons": Descriptors.NumRadicalElectrons(mol),
        "BalabanJ": BalabanJ(mol),
        "BertzCT": BertzCT(mol),
        "MolMR": MolMR(mol),
        "NumAtomStereoCenters": CalcNumAtomStereoCenters(mol),
        "NumUnspecifiedAtomStereoCenters": CalcNumUnspecifiedAtomStereoCenters(mol),
        "NumSpiroAtoms": CalcNumSpiroAtoms(mol),
        "NumBridgeheadAtoms": CalcNumBridgeheadAtoms(mol),
        "Chi0n": CalcChi0n(mol),
        "Chi0v": CalcChi0v(mol),
        "Chi1n": CalcChi1n(mol),
        "Chi1v": CalcChi1v(mol),
        "NumSaturatedRings": NumSaturatedRings(mol),
        "NumSaturatedCarbocycles": NumSaturatedCarbocycles(mol),
        "NumSaturatedHeterocycles": NumSaturatedHeterocycles(mol),
    })

def calculate_morgan_fingerprint(mol, radius=2, n_bits=2048):
    """Generate Morgan fingerprint for a molecule as a bit vector."""
    if mol is None:
        return pd.Series([None])
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius)
    arr = np.zeros((n_bits,), dtype=int)
    Chem.DataStructs.ConvertToNumpyArray(fingerprint, arr)
    return arr



def numpy_to_bitvect(np_array):
    # Convrt numpy array (binary) to an ExplicitBitVect
    bit_vect = DataStructs.CreateFromBitString(''.join(str(int(x)) for x in np_array))
    return bit_vect

# Compute fingerprints list converting each value to a bit vecter
def compute_tanimoto_similarities(df_sample):
    fingerprints = [numpy_to_bitvect(fp) for fp in df_sample['Morgan_f'].values]

    # Initialize a 2D aray to store the Tanimoto similarity values
    n = len(fingerprints)
    similarity_matrix = np.zeros((n, n))

    # Comput the Tanimoto similarity betwen each pair of fingerprints
    for i in range(n):
        for j in range(i, n):  # Only comput for j > i to avoid repetiton
            similarity = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity  # Symmetric matrx

    # Now similarity_matrix contins all pairwise Tanimoto similarities
    # If you need them in a flattened aray (e.g., for furter analysis or plotting)
    similarities = similarity_matrix[np.triu_indices(n, 1)]  # Flatten upper triangle of the matrix

    # Return both the similarities (flattened) and similarity matrix
    return similarities, similarity_matrix


    # Compute fingerprints list converting each value to a bit vecter
def compute_dice_similarities(df_sample):
    fingerprints = df_sample['Morgan_f'].values

    # Initialize a 2D aray to store the Tanimoto similarity values
    n = len(fingerprints)
    similarity_matrix = np.zeros((n, n))

    # Comput the Tanimoto similarity betwen each pair of fingerprints
    for i in range(n):
        for j in range(i, n):  # Only comput for j > i to avoid repetiton
            similarity = dice(fingerprints[i], fingerprints[j])
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity  # Symmetric matrx

    # Now similarity_matrix contins all pairwise Tanimoto similarities
    # If you need them in a flattened aray (e.g., for furter analysis or plotting)
    similarities = similarity_matrix[np.triu_indices(n, 1)]  # Flatten upper triangle of the matrix

    # Return both the similarities (flattened) and similarity matrix
    return similarities, similarity_matrix


