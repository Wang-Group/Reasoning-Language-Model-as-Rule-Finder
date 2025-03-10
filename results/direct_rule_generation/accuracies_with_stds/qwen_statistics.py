from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

import pandas as pd
from rdkit import Chem
import os

import random
random.seed(42)
num_cores = 16

# rules generate by qwen-max-2025-01-25 looking through the paper
def rule2matrix(smiles_list):
    # Define SMARTS patterns for functional groups and structural features
    aromatic_ring = 'c1ccccc1'
    hydroxyl_group = '[OX2H]'
    alkoxy_group = '[OX2][#6]'
    amino_group = '[NX3;H2,H1][#6]'
    nitro_group = '[NX3](=O)=O'
    halogen_group = '[F,Cl,Br,I]'
    trifluoromethyl_group = '[CX4][F][F][F]'
    carboxylic_acid = '[CX3](=O)[OX2H1]'
    carbonyl_group = '[CX3]=[OX1]'
    aldehyde_group = '[CX3H1](=O)'
    bulky_group = '[CX4;!$([CX4][C])][CX4;!$([CX4][C])]'
    long_aliphatic_chain = '[CH2][CH2][CH2][CH2][CH2]'

    # Define the rules with their associated patterns and predictions
    rules = [
        {
            'number': 1,
            'description': 'High Yield: Aromatic ring present.',
            'patterns': [[aromatic_ring]],
            'prediction': 1
        },
        {
            'number': 2,
            'description': 'High Yield: Hydroxyl group present.',
            'patterns': [[hydroxyl_group]],
            'prediction': 1
        },
        {
            'number': 3,
            'description': 'High Yield: Alkoxy group present.',
            'patterns': [[alkoxy_group]],
            'prediction': 1
        },
        {
            'number': 4,
            'description': 'High Yield: Amino group present.',
            'patterns': [[amino_group]],
            'prediction': 1
        },
        {
            'number': 5,
            'description': 'Low Yield: Nitro group present.',
            'patterns': [[nitro_group]],
            'prediction': -1
        },
        {
            'number': 6,
            'description': 'Low Yield: Halogen group present.',
            'patterns': [[halogen_group]],
            'prediction': -1
        },
        {
            'number': 7,
            'description': 'Low Yield: Trifluoromethyl group present.',
            'patterns': [[trifluoromethyl_group]],
            'prediction': -1
        },
        {
            'number': 8,
            'description': 'Low Yield: Bulky functional groups present.',
            'patterns': [[bulky_group]],
            'prediction': -1
        },
        {
            'number': 9,
            'description': 'Low Yield: Long aliphatic chains present.',
            'patterns': [[long_aliphatic_chain]],
            'prediction': -1
        },
        {
            'number': 10,
            'description': 'High Yield: Carboxylic acid group present.',
            'patterns': [[carboxylic_acid]],
            'prediction': 1
        },
        {
            'number': 11,
            'description': 'Low Yield: Carbonyl or aldehyde group present.',
            'patterns': [[carbonyl_group, aldehyde_group]],
            'prediction': -1
        }
    ]

    # Compile SMARTS patterns
    for rule in rules:
        compiled_patterns = []
        for group in rule.get('patterns', []):
            compiled_group = [Chem.MolFromSmarts(p) for p in group]
            compiled_patterns.append(compiled_group)
        rule['compiled_patterns'] = compiled_patterns

    # Initialize results list
    results = []

    # Process each SMILES string
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            # If the molecule cannot be parsed, append a row of zeros
            results.append([0] * len(rules))
            continue
        row = []
        for rule in rules:
            try:
                match = False
                # Check required patterns
                for compiled_group in rule['compiled_patterns']:
                    group_match = False
                    for pat in compiled_group:
                        if mol.HasSubstructMatch(pat):
                            group_match = True
                            break
                    if not group_match:
                        match = False
                        break
                    match = True
                if match:
                    row.append(rule['prediction'])
                else:
                    row.append(0)
            except Exception as e:
                # In case of any error, append 0
                row.append(0)
        results.append(row)

    # Create DataFrame with results
    df = pd.DataFrame(results, columns=[f'Rule {rule['number']}' for rule in rules])
    return df

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

feature_df = rule2matrix(list(dataset['SMILES']))

# Initialize the model
label = df[target].values

exp_feature = dataset[['Fe_loading','modifier/SBU']]
feature = pd.concat([feature_df,exp_feature],axis=1)
feature = feature.values
loo_accuracies = []
loo = LeaveOneOut()

model_rf = RandomForestClassifier(n_estimators=100, max_depth=4,random_state=42, n_jobs=num_cores)
rfecv = RFECV(estimator=model_rf, step=1, cv=loo, scoring='accuracy', n_jobs=num_cores)  # 5-fold CV for feature selection
rfecv.fit(feature, label)
# Get the selected features
selected_features = rfecv.support_  # Boolean mask of selected features
print(f"Number of selected features: {sum(selected_features)}")
print(f"Selected features: {np.where(selected_features)[0]}")  # Indices of selected features
# Reduce the feature set to the selected ones
feature = feature[:, selected_features]
# Initialize Leave-One-Out cross-validator
loo = LeaveOneOut()

if feature.shape[1] <= 4:
    depth = feature.shape[1]
else:
    depth = 4



num_seeds = 50
# Sample 50 random seeds from [0, 10000]
random_seeds = random.sample(range(0, 10001), num_seeds)
loo_accuracies_all_seeds = []

# Run LOO with the sampled random seeds
for seed in random_seeds:
    # Initialize the RandomForestClassifier with a specific random seed
    model_cla = RandomForestClassifier(n_estimators=500, max_depth=depth, max_features=None,random_state=seed, n_jobs=num_cores)
    
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