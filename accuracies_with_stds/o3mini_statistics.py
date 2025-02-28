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


# rules generate by o3-mini-high after looking through the paper
def rule2matrix(smiles_list):
    # Define SMARTS patterns
    # Carboxylic acid group
    carboxylic_acid_patterns = ['[CX3](=O)[OX2H1]', '[CX3](=O)[O-]']

    # Thiol group (-SH)
    thiol_pattern = '[SX2H]'

    # Aldehyde group (-CHO)
    aldehyde_pattern = '[$([CX3H][#6]),$([CX3H2])]=[OX1]'
    
    # Ether linkage in aliphatic chains
    ether_linkage_pattern = '[#6][OX2][#6]'

    # Halogens
    halogen_pattern = '[F,Cl,Br,I]'
    
    # Nitro group
    nitro_group_pattern = '[NX3](=O)[O-]'
    
    # Aromatic ring with halogen or nitro substituents
    halogen_on_aromatic_pattern = '[c][F,Cl,Br,I]'
    nitro_on_aromatic_pattern = '[c][NX3](=O)[O-]'

    # Primary amine (-NH2)
    primary_amine_pattern = '[NX3H2]'

    # Aromatic ring pattern
    aromatic_ring_pattern = '[a]1[a][a][a][a][a]1'
    
    # Define the rules with their associated patterns and predictions
    rules = [
        {
            'number': 1,
            'description': 'Low yield (-1): Modifiers containing a thiol group (-SH).',
            'patterns': [
                [thiol_pattern]
            ],
            'prediction': -1
        },
        {
            'number': 2,
            'description': 'Low yield (-1): Modifiers containing an aldehyde group (-CHO).',
            'patterns': [
                [aldehyde_pattern]
            ],
            'prediction': -1
        },
        {
            'number': 3,
            'description': 'Low yield (-1): Modifiers containing polyether-type structures (chains of ethoxy units).',
            'count_function': lambda mol: len(mol.GetSubstructMatches(Chem.MolFromSmarts(ether_linkage_pattern))) >= 3,
            'prediction': -1
        },
        {
            'number': 4,
            'description': 'High yield (+1): Modifiers that are simple carboxylic acids, whether aliphatic or aromatic with modest substituents (e.g., halogen or nitro groups).',
            'patterns': [
                carboxylic_acid_patterns
            ],
            'additional_patterns': [
                [halogen_on_aromatic_pattern, nitro_on_aromatic_pattern]
            ],
            'prediction': 1
        },
        {
            'number': 5,
            'description': 'High yield (+1): Modifiers that are amino acid derivatives (contain both amine and carboxylic acid groups).',
            'patterns': [
                carboxylic_acid_patterns,
                [primary_amine_pattern]
            ],
            'exclude_patterns': [
                [aromatic_ring_pattern]
            ],
            'prediction': 1
        }
    ]
    
    # Compile SMARTS patterns
    for rule in rules:
        if 'patterns' in rule:
            compiled_patterns = []
            for group in rule['patterns']:
                compiled_group = [Chem.MolFromSmarts(p) for p in group]
                compiled_patterns.append(compiled_group)
            rule['compiled_patterns'] = compiled_patterns
        if 'additional_patterns' in rule:
            compiled_additional = []
            for group in rule['additional_patterns']:
                compiled_group = [Chem.MolFromSmarts(p) for p in group]
                compiled_additional.append(compiled_group)
            rule['compiled_additional_patterns'] = compiled_additional
        if 'exclude_patterns' in rule:
            compiled_excludes = []
            for group in rule['exclude_patterns']:
                compiled_group = [Chem.MolFromSmarts(p) for p in group]
                compiled_excludes.append(compiled_group)
            rule['compiled_exclude_patterns'] = compiled_excludes
    
    # Initialize results list
    results = []
    
    # Process each SMILES string
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            # If the molecule cannot be parsed, append a row of zeros
            results.append([0]*len(rules))
            continue
        row = []
        for rule in rules:
            try:
                match = False
                # For rules with count_function (Rule 3)
                if 'count_function' in rule:
                    if rule['count_function'](mol):
                        match = True
                else:
                    # Check required patterns
                    match = True
                    for compiled_group in rule.get('compiled_patterns', []):
                        group_match = False
                        for pat in compiled_group:
                            if mol.HasSubstructMatch(pat):
                                group_match = True
                                break
                        if not group_match:
                            match = False
                            break
                    # Check additional patterns if any (used in Rule 4)
                    if match and 'compiled_additional_patterns' in rule:
                        additional_match = False
                        for compiled_group in rule['compiled_additional_patterns']:
                            for pat in compiled_group:
                                if mol.HasSubstructMatch(pat):
                                    additional_match = True
                                    break
                            if additional_match:
                                break
                        if not additional_match:
                            match = False
                    # Check exclude patterns if any
                    if match and 'compiled_exclude_patterns' in rule:
                        for compiled_group in rule['compiled_exclude_patterns']:
                            for pat in compiled_group:
                                if mol.HasSubstructMatch(pat):
                                    match = False
                                    break
                            if not match:
                                break
                if match:
                    row.append(rule['prediction'])
                else:
                    row.append(0)
            except Exception as e:
                # In case of any error, append 0
                row.append(0)
        results.append(row)
    # Create DataFrame with results
    df = pd.DataFrame(results, columns=[f'Rule {rule["number"]}' for rule in rules])
    return df


# Load the target column
target = 'yield'
df = pd.read_csv("/home/lnh/GPT_GE/agent_ml/agent/data/data.csv")[[target]]

# Calculate the median of the target column
median_value = df[target].median()

# Transform the target column into 0/1 based on the median
df[target] = (df[target] > median_value).astype(int)

# Verify the transformation
print(f"Median value of '{target}': {median_value}")
print(df.head())



# Load the dataset
dataset = pd.read_csv("/home/lnh/GPT_GE/agent_ml/agent/data/data.csv")

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