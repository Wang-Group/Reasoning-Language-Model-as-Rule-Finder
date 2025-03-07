from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

import pandas as pd
from rdkit import Chem
import os

import random
random.seed(42)

all_smiles = list(pd.read_csv("../../agent/data/data.csv")['SMILES'])


seed = 42
DIR = '../../yield_o1'
target_name = 'yield'

all_best_features = pd.DataFrame({
    'Index':[x for x in range(1,37)]
})
# _ = 1
for _ in range(1, 37):
    print(f'LOO {_}')
    folder = os.path.join(DIR,str(_))
    code_file = os.path.join(folder,"best_rule_code.txt")
    with open(code_file) as f:
        code = f.read()
    exec(code, globals())
    df = rule2matrix(all_smiles) # type: ignore
    selected_train_matrix = pd.read_csv(os.path.join(folder,'best_train_matrix.csv'))
    selected_test_matrix = pd.read_csv(os.path.join(folder,'best_test_matrix.csv'))
    # # model_cla = ExtraTreesClassifier(n_estimators=500, random_state=seed,n_jobs=64)
    selected_rules = list(selected_train_matrix.columns.values)
    df = df[selected_rules]
    # print(selected_rules)
    rule_title = ['LOO_'+str(_)+"_"+x for x in selected_train_matrix.columns.values]
    df.columns = rule_title
    all_best_features = pd.concat([all_best_features,df],axis=1)


all_best_features = all_best_features.iloc[:,1:]

# Function to find groups of columns with identical values
def find_duplicate_columns(df):
    # Create a dictionary to map unique column values to column names
    value_to_columns = {}
    for col in df.columns:
        col_values = tuple(df[col])  # Convert column values to a tuple (hashable)
        if col_values in value_to_columns:
            value_to_columns[col_values].append(col)
        else:
            value_to_columns[col_values] = [col]

    # Extract groups of column names with identical values
    duplicate_columns = {
        key: value for key, value in value_to_columns.items() if len(value) > 1
    }
    return duplicate_columns

# Apply the function to the all_best_features DataFrame
duplicate_columns = find_duplicate_columns(all_best_features)

# Print groups of duplicate columns
if duplicate_columns:
    print("Groups of columns with identical values:")
    for group in duplicate_columns.values():
        print(group)
else:
    print("No duplicate columns found.")
    

# Function to find groups of columns with identical values
def find_duplicate_columns(df):
    value_to_columns = {}
    for col in df.columns:
        col_values = tuple(df[col])  # Convert column values to a tuple (hashable)
        if col_values in value_to_columns:
            value_to_columns[col_values].append(col)
        else:
            value_to_columns[col_values] = [col]

    # Extract groups of column names with identical values
    duplicate_columns = {
        key: value for key, value in value_to_columns.items() if len(value) > 1
    }
    return duplicate_columns

# Find duplicate columns in the all_best_features DataFrame
duplicate_columns = find_duplicate_columns(all_best_features)

# Load the SHAP data
shap_data = pd.read_csv("../../o1_rules_SHAP_value/mean_shap_values.csv")

# Calculate absolute mean SHAP values
shap_data['Abs_Mean_SHAP_Value'] = shap_data['Mean_SHAP_Value'].abs()

# Sort by absolute mean SHAP value and get the top 20 features
top_20_shap = shap_data.nlargest(20, 'Abs_Mean_SHAP_Value')

# Extract the feature names from the top 20 SHAP values
top_20_features = top_20_shap['Rule'].tolist()

# Check if any top 20 features are in the same group of duplicate columns
overlapping_features = []

for group in duplicate_columns.values():
    overlap = list(set(group) & set(top_20_features))
    if len(overlap) > 1:  # More than one feature from the same group
        overlapping_features.append(overlap)

# Print the results
if overlapping_features:
    print("Groups of top 20 features in the same duplicate group:")
    for group in overlapping_features:
        print(group)
else:
    print("No top 20 features are in the same duplicate group.")
    
    

# Load the target column
target = 'yield'
df = pd.read_csv("../../agent/data/data.csv")[[target]]

# Calculate the median of the target column
median_value = df[target].median()

# Transform the target column into 0/1 based on the median
df[target] = (df[target] > median_value).astype(int)

# Verify the transformation
print(f"Median value of '{target}': {median_value}")
print(df.head())



# Load the dataset
dataset = pd.read_csv("../../agent/data/data.csv")


# Convert the labels to numpy array
label = df[target].values

# Read the feature names back into a list
with open('../../o1_rules_SHAP_value/top_20_shap_features.txt', 'r') as f:
    top_20_features_from_file = [line.strip() for line in f]
# Remove specific items from the list
# top_20_features_from_file = [feature for feature in top_20_features_from_file if feature not in ['LOO_11_Rule 7', 'LOO_27_Rule 8']]
# Select the columns corresponding to the top 20 features
top_20_features_data = all_best_features[top_20_features_from_file]

feature = dataset[['Fe_loading','modifier/SBU']]
feature = pd.concat([feature,top_20_features_data.iloc[:, :4]],axis=1)
feature = feature.values
# Initialize Leave-One-Out cross-validator
loo = LeaveOneOut()

num_seeds = 50
# Sample 50 random seeds from [0, 10000]
random_seeds = random.sample(range(0, 10001), num_seeds)
loo_accuracies_all_seeds = []

# Run LOO with the sampled random seeds
for seed in random_seeds:
    # Initialize the RandomForestClassifier with a specific random seed
    model_cla = RandomForestClassifier(n_estimators=500, max_depth=4, max_features=None,random_state=seed, n_jobs=64)
    
    # List to store accuracies for this seed
    loo_accuracies = []
    
    # Perform LOO cross-validation
    for train_index, test_index in loo.split(feature):
        X_train, X_test = feature[train_index], feature[test_index]
        y_train, y_test = label[train_index], label[test_index]
        
        # Train the model
        model_cla.fit(X_train, y_train)
        
        # Predict on the test set
        y_pred = model_cla.predict(X_test)
        
        # Calculate accuracy for this split
        loo_accuracies.append(accuracy_score(y_test, y_pred))
    
    # Store the mean accuracy for this seed
    loo_accuracies_all_seeds.append(np.mean(loo_accuracies))

# Calculate mean and standard deviation of LOO accuracies across all seeds
mean_loo_accuracy = np.mean(loo_accuracies_all_seeds)
std_loo_accuracy = np.std(loo_accuracies_all_seeds)

# Print the results
print(f"Mean LOO Accuracy over {num_seeds} seeds: {mean_loo_accuracy:.4f}")
print(f"Standard Deviation of LOO Accuracy over {num_seeds} seeds: {std_loo_accuracy:.4f}")