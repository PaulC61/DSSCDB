import pandas as pd
import numpy as np
import dask.dataframe as dd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools

# Initialisation
def find_dye_devices(smiles, dye_df, device_df):
    unique_device_df = device_df.drop_duplicates()

    device_index = list(dye_df.loc[dye_df == smiles].index)
    device_set = device_df.loc[device_index].drop_duplicates()

    enum_device_df = pd.concat([device_set, unique_device_df], axis=0, ignore_index=True).drop_duplicates(keep=False)
    
    enum_device_df["SMILES"] = smiles

    radius = 3
    nBits = 1024

    PandasTools.AddMoleculeColumnToFrame(enum_device_df, smilesCol='SMILES')

    ECFP6 = [AllChem.GetMorganFingerprintAsBitVect(x, radius=radius, nBits=nBits) for x in enum_device_df['ROMol']]

    ecfp6_name = [f"Bit_{i}" for i in range(nBits)]

    ecfp6_bits = [list(l) for l in ECFP6]

    ecfp6_df = pd.DataFrame(ecfp6_bits, columns=ecfp6_name, index=enum_device_df.index)

    enum_device_df = pd.concat([ecfp6_df, enum_device_df.drop(["SMILES", "ROMol"], axis=1)], axis=1)
    
    enum_device_df["PCE"] = -1

    return enum_device_df

def enumerate_devices(device_df, columns, smiles_col):
    dye_device_enum_df = pd.DataFrame(columns=columns)

    for smiles in set(list(smiles_col)):
        dye_device_enum_df = pd.concat([dye_device_enum_df, find_dye_devices(smiles=smiles, dye_df=smiles_col, device_df=device_df)], axis=0, ignore_index=True) 

    return dye_device_enum_df


# selecting samples
# scoring predictions
def prediction_var(rf_model, X_test):

    tree_predictions = np.array([tree.predict(X_test.values) for tree in rf_model.estimators_])
    tree_prediction_var = tree_predictions.var(0)
    X_test["PCE"] = rf_model.predict(X_test)
    X_test["PredVar"] = tree_prediction_var

    return X_test