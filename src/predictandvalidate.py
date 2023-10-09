import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdFingerprintGenerator
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

class RegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(RegressionModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, output_size)
        )
    
    def forward(self, x):
        return self.model(x)

def smiles_to_morgan(smiles_str, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles_str)
    if mol is not None:
        morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        return np.array(morgan_fp)
    else:
        return np.zeros(nBits)

def get_topological_torsion_fingerprint(smiles_str):
    molecule = Chem.MolFromSmiles(smiles_str)
    fingerprint_generator = rdFingerprintGenerator.GetTopologicalTorsionGenerator()
    fingerprint = fingerprint_generator.GetFingerprint(molecule)
    return np.array(fingerprint)

def denormalize(scaled_value, min_value, max_value):
    return scaled_value * (max_value - min_value) + min_value

# Load train data to get min-max values for denormalization
train_data = pd.read_csv('train.csv')
min_values = train_data.iloc[:, 1:].min().values
max_values = train_data.iloc[:, 1:].max().values

# Load test data and ground truth
test_data = pd.read_csv('test.csv')
test_w_gt_data = pd.read_csv('test_w_gt.csv')

# Load the saved model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_targets = len(test_data.columns) - 1  # Subtracting 1 for the SMILES column
model = RegressionModel(input_size=4096, output_size=num_targets)
model.load_state_dict(torch.load('regressionmodel2.pth'))
model.to(device)
model.eval()

# Predict scores and denormalize
predicted_scores_list = []
for smile in test_data["SMILES"]:
    morgan_fp = smiles_to_morgan(smile)
    torsion_fp = get_topological_torsion_fingerprint(smile)
    concatenated_fp = np.concatenate([morgan_fp, torsion_fp])
    
    fp_tensor = torch.tensor([concatenated_fp], dtype=torch.float32).to(device)
    
    with torch.no_grad():
        predicted_scores = model(fp_tensor)
    denormalized_scores = denormalize(predicted_scores.cpu().numpy()[0], min_values, max_values)
    predicted_scores_list.append(denormalized_scores)

# Convert denormalized scores to DataFrame
predicted_scores_df = pd.DataFrame(predicted_scores_list, columns=test_data.columns[1:])
predicted_scores_df.insert(0, "SMILES", test_data["SMILES"])

predicted_scores_df.to_csv('test_predicted.csv', index=False)

# Compute Mean Squared Error against ground truth
mse = mean_squared_error(test_w_gt_data.iloc[:, 1:].values, predicted_scores_df.iloc[:, 1:].values)
mae = mean_absolute_error(test_w_gt_data.iloc[:, 1:].values, predicted_scores_df.iloc[:, 1:].values)
print('Mean Squared Error: ', mse)
print('Mean Absolute Error: ', mae)
