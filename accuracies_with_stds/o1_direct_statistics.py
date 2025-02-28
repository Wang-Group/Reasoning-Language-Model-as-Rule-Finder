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

def rule2matrix(smiles_list):
    # Import necessary modules
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import AllChem

    # Define SMARTS patterns for functional groups

    # Carboxylic acid group (-COOH)
    carboxylic_acid = '[CX3](=O)[OX1H0-,OX2H1]'

    # Benzoic acid group
    benzoic_acid = 'c1ccccc1C(=O)[O;H1,-]'

    # Halogen attached to aromatic ring
    halogen_on_aromatic = '[c][F,Cl,Br,I]'

    # Nitro group attached to aromatic ring
    nitro_on_aromatic = '[c][N+](=O)[O-]'

    # Amino group attached to aromatic ring
    amino_on_aromatic = '[c][NH2]'

    # Aldehyde group attached to aromatic ring
    aldehyde_on_aromatic = '[c][CH=O]'

    # Thiol group attached to aromatic ring
    thiol_on_aromatic = '[c][SH]'

    # Pattern for aliphatic chain of five carbons
    aliphatic_chain_five_carbons = '[CH2][CH2][CH2][CH2][CH3]'

    # Thiol group in aliphatic chain (e.g., 3-mercaptopropionic acid)
    aliphatic_thiol = '[#6][#6][SX2H]'

    # PEG-like chain (multiple ether linkages)
    peg_chain = '[#6][OX2][#6][OX2][#6][OX2][#6]'

    # Amino acid backbone pattern
    amino_acid = '[NX3;H2][CX4][CX3](=O)[O;H1,-]'

    # Phenolate group (deprotonated phenol)
    phenolate = '[c][O-]'

    # Define the rules
    rules = [
        {
            'number': 1,
            'description': 'High Yield: Modifiers containing para-substituted benzoic acids with substituents –Br, –NO₂, –NH₂ at para position.',
            'patterns': [
                [benzoic_acid],  # Benzoic acid
                [halogen_on_aromatic, nitro_on_aromatic, amino_on_aromatic]  # Substituents
            ],
            'prediction': 1
        },
        {
            'number': 2,
            'description': 'Low Yield: Modifiers containing para-substituted benzoic acids with substituents –CHO, –SH at para position.',
            'patterns': [
                [benzoic_acid],  # Benzoic acid
                [aldehyde_on_aromatic, thiol_on_aromatic]  # Substituents
            ],
            'prediction': -1
        },
        {
            'number': 3,
            'description': 'High Yield: Simple aliphatic carboxylic acids without additional polar functional groups near the metal site.',
            'patterns': [
                [carboxylic_acid],  # Carboxylic acid group
                [aliphatic_chain_five_carbons]  # Aliphatic chain of five carbons
            ],
            'exclude_patterns': [
                ['[NX3]', '[OX2]', '[SX2]']  # Exclude N, O, S (excluding the carboxylic acid oxygen)
            ],
            'prediction': 1
        },
        {
            'number': 4,
            'description': 'High Yield: Modifiers with thiol groups (-SH) far from the carboxyl group (e.g., in aliphatic chain).',
            'patterns': [
                [carboxylic_acid],  # Carboxylic acid
                [aliphatic_thiol]  # Thiol group in aliphatic chain
            ],
            'prediction': 1
        },
        {
            'number': 5,
            'description': 'Low Yield: Modifiers with multiple polar functional groups or strongly coordinating moieties (e.g., PEG-like chains).',
            'patterns': [
                [carboxylic_acid],  # Carboxylic acid
                [peg_chain]  # PEG-like chain
            ],
            'prediction': -1
        },
        {
            'number': 6,
            'description': 'High Yield: Modifiers are amino acids or related molecules (e.g., aspartic acid).',
            'patterns': [
                [carboxylic_acid],  # Carboxylic acid
                [amino_acid]  # Amino acid backbone
            ],
            'prediction': 1
        },
        {
            'number': 7,
            'description': 'Low Yield: Modifiers that directly coordinate Fe (e.g., phenolate, –SH on aromatic ring, –CHO).',
            'patterns': [
                [carboxylic_acid],
                [phenolate, thiol_on_aromatic, aldehyde_on_aromatic]  # Strong Fe-coordinating groups
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
        # Compile exclude patterns if any
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
                match = True
                # Check exclude patterns if any
                if 'exclude_patterns' in rule:
                    for group in rule['compiled_exclude_patterns']:
                        for pat in group:
                            if mol.HasSubstructMatch(pat):
                                match = False
                                break
                        if not match:
                            break
                    if not match:
                        row.append(0)
                        continue
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
                if match:
                    row.append(rule['prediction'])
                else:
                    row.append(0)
            except Exception:
                # In case of any error, append 0
                row.append(0)
        results.append(row)
    # Create DataFrame with results
    df = pd.DataFrame(results, columns=[f'Rule {rule['number']}' for rule in rules])
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