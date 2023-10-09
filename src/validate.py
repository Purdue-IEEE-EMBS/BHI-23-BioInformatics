import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# Load ground truth and predicted data
test_w_gt_data = pd.read_csv('test_w_gt.csv')
predicted_scores_df = pd.read_csv('test_predicted.csv')

test_w_gt_data = test_w_gt_data.sort_values(by="SMILES").reset_index(drop=True)
predicted_scores_df = predicted_scores_df.sort_values(by="SMILES").reset_index(drop=True)

# Compute Mean Absolute Error against ground truth
mae = mean_absolute_error(test_w_gt_data.iloc[:, 1:].values, predicted_scores_df.iloc[:, 1:].values)
mse = mean_squared_error(test_w_gt_data.iloc[:, 1:].values, predicted_scores_df.iloc[:, 1:].values)

print("Mean Squared Error: ", mse)
print("Mean Absolute Error:", mae)

