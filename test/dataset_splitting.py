# Splitting dataset for iterative GPT pipeline
from agent.state import OverallState

target_name = "yield"
Fe_loading_flag = False
INPUT = {
  "target_name": target_name,
}
state = OverallState(INPUT)
print(state)

import os
from datetime import datetime
import pandas as pd
from sklearn.model_selection import LeaveOneOut
import pkg_resources

# GPT_model = state['GPT_model']
target_name = state['target_name']
# Reading the dataset
dataset_path = pkg_resources.resource_filename('agent.data', 'data.csv')
data = pd.read_csv(dataset_path)
name = pd.read_csv(pkg_resources.resource_filename('agent.data', 'name.csv'))['name']
# Check and create appropriate `append_name` based on `target_name`
if target_name == 'Fe/Hf':
    append_name = 'Fe_Hf'
elif target_name == 'modifier/SBU':
    append_name = 'modi_SBU'
elif target_name == 'yield':
    append_name = 'yield'
else:
    raise ValueError(f"Invalid target_name: {target_name}. Must be 'Fe/Hf', 'modifier/SBU' or 'yield'.")

# Initialize the output folder
date_index = 'yield_o1'
output_folder = append_name + '_' + date_index
os.makedirs(output_folder, exist_ok=True)


# Load SMILES, name anc experimental descriptors for samples
SMILES = data['SMILES']
NAME = name
Fe_loading = data['Fe_loading']
modifier_SBU = data['modifier/SBU']
Fe_Hf = data['Fe/Hf']
target_values = data[target_name]
high_or_low_loading = data[target_name] > data[target_name].median()

# Generate dataset containing necessary columns
dataset = pd.DataFrame({
    target_name:target_values,
    'name': NAME,
    'SMILES': SMILES,
    f'{target_name}_high_or_low_value': high_or_low_loading
})

exp_dataset = pd.DataFrame({
    'Fe_loading': Fe_loading,
    'modifier/SBU': modifier_SBU,
    'Fe/Hf': Fe_Hf
})

# Add Fe loading values to the dataset if commanded
if Fe_loading_flag:
    Fe_loading_df = pd.read_csv(pkg_resources.resource_filename('agent.data', 'data_yield.csv'))[['yield_high_or_low_pred_by_Fe_loading']]
    Fe_loading_data = pd.read_csv(pkg_resources.resource_filename('agent.data', 'data.csv'))[['Fe_loading']]
    Fe_loading_df = pd.concat([Fe_loading_df, Fe_loading_data],axis=1)
    
# Read the reference values for rule metrics (support, confidence, lift and leverage) and ML metrics (accuracies and SHAP values).
rule_hist_path = pkg_resources.resource_filename('agent.data', f"pre_rule_metric_{append_name}.txt")
with open(rule_hist_path,'r') as f:
    content = f.read()
content = '< Previous Rule Metrics for Reference: >\n'+ content + '\n< Rule Metrics During the Iteration of This Program: >\n'
ml_hist_path = pkg_resources.resource_filename('agent.data', f"pre_ML_metric_{append_name}.txt")
with open(ml_hist_path,'r') as f:
    content1 = f.read()
content1 = '< Previous Accuracy for Reference: >\n'+ content1 + '\n< Accuracy and SHAP During the Iteration of This Program: >\n'
# Leave-One-Out cross-validation
loo = LeaveOneOut()
index = 1

# Generate LOO splitted datasets
for train_index, test_index in loo.split(dataset):
    if index == 36:
        # Get the train and test sets
        train_set = dataset.iloc[train_index]
        test_set = dataset.iloc[test_index]
        
        exp_train = exp_dataset.iloc[train_index]
        exp_test = exp_dataset.iloc[test_index]
        
        
        Fe_loading_pred_train = Fe_loading_df.iloc[train_index]
        Fe_loading_pred_test = Fe_loading_df.iloc[test_index]
        
        # Create a new directory for each split
        split_folder = os.path.join(output_folder, str(index))
        os.makedirs(split_folder, exist_ok=True)
        
        # Save the train and test sets
        train_set.to_csv(os.path.join(split_folder, 'train_set.csv'), index=False)
        test_set.to_csv(os.path.join(split_folder, 'test_set.csv'), index=False)
        
        exp_train.to_csv(os.path.join(split_folder, 'exp_train.csv'), index=False)
        exp_test.to_csv(os.path.join(split_folder, 'exp_test.csv'), index=False)
        
        Fe_loading_pred_train.to_csv(os.path.join(split_folder, 'Fe_pred_train.csv'), index=False)
        Fe_loading_pred_test.to_csv(os.path.join(split_folder, 'Fe_pred_test.csv'), index=False)
        
        with open(os.path.join(split_folder,'rule_metric_log.txt'),'w') as f1:
            f1.write(content)
        
        with open(os.path.join(split_folder,'ML_metric_log.txt'),'w') as f1:
            f1.write(content1)
        
    index += 1
print(f"Data splits have been saved in the '{output_folder}' directory.")
# print('This inputs containing experimental data')