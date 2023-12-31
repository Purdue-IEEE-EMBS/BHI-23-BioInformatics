import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from rdkit.Chem import rdFingerprintGenerator

# function to turn SMILES into morgan thingy
def smiles_to_morgan(smiles_str, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles_str)
    if mol is not None:
        morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        return np.array(morgan_fp)
    else:
        return np.zeros(nBits)
    
# function to turn SMILES into topological torsion fingerprint
def get_topological_torsion_fingerprint(smiles_str):
  molecule = Chem.MolFromSmiles(smiles_str)
  fingerprint_generator = rdFingerprintGenerator.GetTopologicalTorsionGenerator()
  fingerprint = fingerprint_generator.GetFingerprint(molecule)

  return np.array(fingerprint)

print("Loading data...")
data = pd.read_csv('train.csv')

# Convert SMILES to Morgan fingerprints
print("Converting SMILES to Fingerprints...")
data_X = np.array([np.concatenate((smiles_to_morgan(smile), get_topological_torsion_fingerprint(smile))) for smile in tqdm(data["SMILES"], desc="Converting to 4096 len Vector")])

# regularize w min max scaling
print("Regularizing docking scores...")
scaler = MinMaxScaler()
data_Y_values = []
for col in tqdm(data.columns[1:], desc="Regularizing"):
    column_data = data[col].values.reshape(-1, 1)
    scaled_column = scaler.fit_transform(column_data)
    data_Y_values.append(scaled_column)
data_Y = np.hstack(data_Y_values)

# Save preprocessed data 
print("Saving preprocessed data...")
np.save('data_X.npy', data_X)
np.save('data_Y.npy', data_Y)
print("Data preprocessing complete. Files saved as data_X.npy and data_Y.npy.")
