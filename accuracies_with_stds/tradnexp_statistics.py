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


# Convert the labels to numpy array
label = df[target].values

feature_file = "/home/lnh/GPT_GE/gpt-featurization/result/data/del_and_exp_features.csv"
selected_features = ['Fe_loading','Solvent_area','Fun_LogP_Max']

feature = pd.read_csv(feature_file)[selected_features]
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
    # model_cla = RandomForestClassifier(n_estimators=500, max_depth=4, random_state=seed, n_jobs=64)
    model_cla = RandomForestClassifier(n_estimators=500, max_depth=3, max_features=None,random_state=seed, n_jobs=64)
    
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