import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem

DESCRIPTOR_LIST = Descriptors._descList  # list of (name, function)
DESC_NAMES = [name for name, _ in DESCRIPTOR_LIST]

def rdkit_descriptors_from_smiles(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {name: np.nan for name in DESC_NAMES}
    return {name: func(mol) for name, func in DESCRIPTOR_LIST}

def morgan_bits(mol, radius=2, nBits=2048):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
    arr = np.zeros((nBits,), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def maccs_bits(mol):
    fp = MACCSkeys.GenMACCSKeys(mol)  # 167 bits (index 0..166)
    arr = np.zeros((len(fp),), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def fingerprints_from_smiles(smiles: str, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return (np.full((nBits,), 0, dtype=np.int8),
                np.full((167,), 0, dtype=np.int8))
    return morgan_bits(mol, radius=radius, nBits=nBits), maccs_bits(mol)


def featurize_dataframe(df: pd.DataFrame, smiles_col="SMILES", radius=2, nBits=2048) -> pd.DataFrame:
    # 1) Descriptors
    desc_rows = [rdkit_descriptors_from_smiles(s) for s in df[smiles_col]]
    desc_df = pd.DataFrame(desc_rows)

    # 2) Fingerprints
    morgan_list = []
    maccs_list = []
    for s in df[smiles_col]:
        morgan_arr, maccs_arr = fingerprints_from_smiles(s, radius=radius, nBits=nBits)
        morgan_list.append(morgan_arr)
        maccs_list.append(maccs_arr)

    morgan_df = pd.DataFrame(np.vstack(morgan_list), columns=[f"ECFP_{i}" for i in range(nBits)])
    maccs_df  = pd.DataFrame(np.vstack(maccs_list),  columns=[f"MACCS_{i}" for i in range(167)])

    # zusammenführen (SMILES behalten, falls ihr später noch was wollt)
    out = pd.concat([df.reset_index(drop=True), desc_df, morgan_df, maccs_df], axis=1)
    return out


