from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem import rdMolDescriptors
import numpy as np

def smiles_to_mol(smiles):
    """Convert SMILES string to RDKit Mol object."""
    try:
        return Chem.MolFromSmiles(smiles)
    except:
        return None


def calculate_descriptors(mol):
    """Calculate RDKit molecular descriptors for a given molecule."""
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
    #pd.Series({
        #"MolWt": Descriptors.MolWt(mol),
        #"LogP": Descriptors.MolLogP(mol),
        #"NumHDonors": Descriptors.NumHDonors(mol),
        #"NumHAcceptors": Descriptors.NumHAcceptors(mol),
        #"TPSA": Descriptors.TPSA(mol)
    #})


def calculate_morgan_fingerprint(mol, radius=2, n_bits=2048):
    """Generate Morgan fingerprint for a molecule as a bit vector."""
    if mol is None:
        return pd.Series([None])
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius)
    arr = np.zeros((n_bits,), dtype=int)
    Chem.DataStructs.ConvertToNumpyArray(fingerprint, arr)
    return arr
