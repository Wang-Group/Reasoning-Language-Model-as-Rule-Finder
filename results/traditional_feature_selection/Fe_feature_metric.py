import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import json
import pickle

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import LeavePOut
from sklearn.model_selection import LeaveOneOut
def load_data(feature_file, target_file,target_name):
    '''Load feature matrix and target values'''
    X = pd.read_csv(feature_file)
    y = pd.read_csv(target_file)[target_name]
    return X, y

def train_random_forest(X_train, y_train, seed):
    '''Train a random forest model with selected seed'''
    dt_classifier = RandomForestClassifier(random_state=seed,max_depth=len(X_train.columns),n_jobs=-1)
    dt_classifier.fit(X_train, y_train)
    return dt_classifier

def evaluate_combinations(X, y, feature_combinations):
    '''Evaluate the performance of different feature combinations using the input feature matrix and corresponding labels.'''
    results = {}
    
    for features in feature_combinations:
        selected_features = list(features)
        X_selected = X[selected_features]

        loo = LeaveOneOut()
        y_true_all = []
        y_pred_all = []

        for train_index, test_index in loo.split(X_selected):
            X_train, X_test = X_selected.iloc[train_index], X_selected.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            for random_seed in [20, 30, 42, 50, 60, 80, 100, 200, 500]:
                dt_classifier = train_random_forest(X_train, y_train, seed=random_seed)
                y_pred = dt_classifier.predict(X_test)
                y_true_all.extend(y_test.values)
                y_pred_all.extend(y_pred)

        LOO_test_accuracy = accuracy_score(y_true_all, y_pred_all)
        results[features] = LOO_test_accuracy
        print(f"Evaluated features: {features}, Accuracy: {LOO_test_accuracy}")

    return results



with open('/home/lnh/GPT_GE/gpt-featurization/result/output/Fe-Hf_cla/result.pkl', 'rb') as file:
    result = pickle.load(file)
feature_file = "/home/lnh/GPT_GE/gpt-featurization/result/data/del_features.csv"
target_file = "/home/lnh/GPT_GE/gpt-featurization/result/data/all_targets.csv"
target_name = 'Fe/Hf_cla'
X, y = load_data(feature_file, target_file,target_name)      
num_folds=50


lpo = LeavePOut(4)
temperate_split = []
for train_index, test_index in lpo.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    if np.sum(y_test) == 2: # make sure there are two 0s and two 1s in the test set 
        lst = [train_index,test_index]
        temperate_split.append(lst)
samples = np.linspace(start=0,stop=len(temperate_split)-1,num=num_folds)

feature_statistics = {}
test_ids = {}
feature_test_statistics = {}
for iteration,sample in enumerate(samples):
    
    train_index, test_index = temperate_split[int(sample)]
    print(f"Iteration {iteration}: Points left out - {test_index}")
    lst = list(result[iteration].keys())
    feature_lst = [feature for feature in lst[9]]
    for k in range(len(feature_lst)):
        print("Use {} features".format(k+1))
        selected_feature = feature_lst[:k+1]
        X_selected = X[feature_lst]
    
        X_train, X_test = X_selected.iloc[train_index], X_selected.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        test_accuracy = []
        for random_seed in [20,30,42,50,60,80,100,200,500]:
            dt_classifier = train_random_forest(X_train, y_train, seed=random_seed)  # Use a fixed seed for reproducibility
            y_pred = dt_classifier.predict(X_test)
            current_accuracy = accuracy_score(y_test, y_pred)
            y_train_pred = dt_classifier.predict(X_train)
            train_accuracy = accuracy_score(y_train,y_train_pred)
            print("Seed {} train Accuracy:{}".format(random_seed,train_accuracy),end="; ")
            print("Seed {} test Accuracy:{}".format(random_seed,current_accuracy))
            test_accuracy.append(current_accuracy)
        mean_test_accuracy = np.mean(test_accuracy)
        print("{} features mean accuracy:{}".format(k+1,mean_test_accuracy))
        if mean_test_accuracy >= 0.75:
            features = ", ".join(feature_lst)
            test_id = ", ".join([str(x) for x in temperate_split[int(sample)][1]])
            feature_test_combo = (features, test_id)
            feature_test_statistics[feature_test_combo] = feature_test_statistics.get(feature_test_combo, 0) + 1

feature_test_statistics = dict(sorted(feature_test_statistics.items(), key=lambda x: x[0], reverse=True))

# Extract feature combinations
feature_combinations_set = set()
for key in feature_test_statistics.keys():
    features_str = key[0]
    features_list = features_str.split(', ')
    for i in range(1, 6):
        feature_combinations_set.add(tuple(features_list[:i]))

results = evaluate_combinations(X, y, feature_combinations_set)
results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

# Outputs the feature combinations with considerable accuracies (>=0.75) on its split of dataset.
print(results)
