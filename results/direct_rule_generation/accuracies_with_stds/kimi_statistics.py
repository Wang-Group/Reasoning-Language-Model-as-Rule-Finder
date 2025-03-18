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


# rules generate by Kimi-k1.5 looking through the paper
def rule2matrix(smiles_list):
    # Define SMARTS patterns for functional groups
    # Aromatic ring
    aromatic_ring = '[a]'

    # Electron-Withdrawing Groups (EWGs)
    nitro_group = '[NX3](=O)=O'
    halogen = '[F,Cl,Br,I]'
    trifluoromethyl = '[CX4]([F])([F])[F]'
    
    # EWG attached to aromatic ring
    ewg_on_aromatic = ['[a][NX3](=O)=O', '[a][F,Cl,Br,I]', '[a][CX4]([F])([F])[F]']
    
    # Carboxylic acid group
    carboxylic_acid = '[CX3](=O)[OX2H1]'
    
    # Electron-Donating Groups (EDGs)
    hydroxyl_group = '[OX2H]'
    amino_group = '[NX3H2]'
    methoxy_group = '[OX2][CH3]'
    edg_on_aromatic = ['[a][OX2H]', '[a][NX3H2]', '[a][OX2][CH3]']

    # Aldehyde group
    aldehyde = '[CX3H][OX1]'

    # Define the rules
    rules = [
        {
            'number': 1,
            'description': 'High yield (+1): Modifiers containing aromatic rings with electron-withdrawing groups attached and connected to a carboxylic acid group.',
            'patterns': [
                ewg_on_aromatic,  # EWG on aromatic ring
                [carboxylic_acid]  # Carboxylic acid group
            ],
            'prediction': 1
        },
        {
            'number': 2,
            'description': 'High yield (+1): Modifiers containing more than one carboxylic acid group.',
            'patterns': [
                [carboxylic_acid]
            ],
            'count_threshold': { carboxylic_acid: 2 },
            'prediction': 1
        },
        {
            'number': 3,
            'description': 'Low yield (-1): Modifiers containing electron-donating groups connected to aromatic rings along with a carboxylic acid group.',
            'patterns': [
                edg_on_aromatic,  # EDG on aromatic ring
                [carboxylic_acid]
            ],
            'prediction': -1
        },
        {
            'number': 4,
            'description': 'Low yield (-1): Modifiers that are bulky or highly branched structures (e.g., having more than one ring).',
            'patterns': [],
            'check_ring_count': 2,
            'prediction': -1
        },
        {
            'number': 5,
            'description': 'Low yield (-1): Modifiers containing aldehyde group along with carboxylic acid group.',
            'patterns': [
                [aldehyde],
                [carboxylic_acid]
            ],
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
    
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            results.append([0]*len(rules))
            continue
        row = []
        try:
            ring_count = mol.GetRingInfo().NumRings()
        except:
            ring_count = 0
        for rule in rules:
            match = True
            # Check for ring count if specified in rule
            if 'check_ring_count' in rule:
                if ring_count >= rule['check_ring_count']:
                    row.append(rule['prediction'])
                else:
                    row.append(0)
                continue
            # Check required patterns
            for compiled_group in rule['compiled_patterns']:
                group_match = False
                for pat in compiled_group:
                    if pat is None:
                        continue
                    matches = mol.GetSubstructMatches(pat)
                    if matches:
                        # If a count threshold is specified for this pattern
                        if 'count_threshold' in rule and pat in [Chem.MolFromSmarts(k) for k in rule['count_threshold']]:
                            smarts = Chem.MolToSmarts(pat)
                            threshold = rule['count_threshold'][smarts]
                            if len(matches) >= threshold:
                                group_match = True
                                break
                        else:
                            group_match = True
                            break
                if not group_match:
                    match = False
                    break
            if match:
                row.append(rule['prediction'])
            else:
                row.append(0)
        results.append(row)
    # Create DataFrame with results
    df = pd.DataFrame(results, columns=[f"Rule {rule['number']}" for rule in rules])
    return df


# Load the target column
target = 'yield'
df = pd.read_csv("../../../agent/data/data.csv")[[target]]

# Calculate the median of the target column
median_value = df[target].median()

# Transform the target column into 0/1 based on the median
df[target] = (df[target] > median_value).astype(int)

# Verify the transformation
print(f"Median value of '{target}': {median_value}")
print(df.head())



# Load the dataset
dataset = pd.read_csv("../../../agent/data/data.csv")

feature_df = rule2matrix(list(dataset['SMILES']))

# Initialize the model
label = df[target].values

exp_feature = dataset[['Fe_loading','modifier/SBU','Fe/Hf']]
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