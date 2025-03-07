import os
import time

import numpy as np
import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
import shap

from agent.state import AgentState,BaseMessage
from agent.client import client


def high_low(high_or_low):
    if high_or_low:
        return 'high'
    else:
        return 'low'
def rules_feedback(data, predictions, val_performances, shap_expressions,tar_column_name):
    """Check samples one by one for whether the prediction and SHAP value is consistent with the true label.
    Args:
        data: list, the train set ranged in corresponding to the validation samples.
        predictions: list, model predictions on the validation set.
        val_performances: list, the accuracy on the validation set.
        shap_expressions: list, the analysis of every SHAP value of rules and corresponding label of the sample.
        tar_column_name: list, the column name of the target column.
    """
    
    # This should apply the rules to your data and return prediction and actual yields
    differences = [f"For molecule {data['SMILES'].values[i]}, the predicted {tar_column_name} is {high_low(predictions[i])} and the experimental value is {high_low(data[tar_column_name].values[i])} \n  {shap_expressions[i]} \n"  for i in range(len(predictions))]
    feedback = f'''< 5 Fold Validation Performance: >\n 
    An accuracy of {val_performances} was obtained using classification model. \n \n \n 
    < SHAP Analysis Feedback > \n'''
    for difference in differences:
        feedback += difference
        feedback += ''
    return feedback



def ML_Calculator(state:AgentState):
    '''
    Calculate  5-fold cross-validation accuracy and SHAP values and generate comparison between SHAP values and predictions rule by rule, sample by sample.
    '''
    try:
        #Calculate accuracy and SHAP on validation set for feedback
        tar_column_name = f'{state.target_name}_high_or_low_value'
        seed = state.seed
        train_set = pd.read_csv(os.path.join(state.output_dir,state.train_file))
        train_features = pd.read_csv(os.path.join(state.output_dir,state.train_matrix))
        test_features = pd.read_csv(os.path.join(state.output_dir,state.test_matrix))
        loo = LeaveOneOut()
        train_labels = train_set[tar_column_name]
        if state.Tran_model == 'ETC':
            estimator = ExtraTreesClassifier(n_estimators=500, random_state=seed,n_jobs=64)
        elif state.Tran_model == 'RFC':
            estimator = RandomForestClassifier(n_estimators=500, random_state=seed,n_jobs=64)

        selector = RFECV(estimator, step=1, cv=loo, scoring='accuracy',n_jobs=64)
        # estimator = RandomForestClassifier(n_estimators=500, random_state=seed,n_jobs=64)
        # cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        # selector = RFECV(estimator, step=1, cv=cv_strategy, scoring='accuracy',n_jobs=64)

        selector.fit(train_features.values, train_labels.values)

        selection = selector.support_
        selected_train_matrix_file = 'selected_train_mtx.csv'
        selected_test_matrix_file = 'selected_test_mtx.csv'
        selected_train_matrix = train_features.iloc[:,selection]
        selected_test_matrix = test_features.iloc[:,selection]

        selected_train_matrix.to_csv(os.path.join(state.output_dir,selected_train_matrix_file),index=False)
        selected_test_matrix.to_csv(os.path.join(state.output_dir,selected_test_matrix_file),index=False)

        all_val_performances = []
        train_indexes = []
        valid_indexes = []
        train_cla_predictions = []
        valid_cla_predictions = []
        shap_expressions_lst = []


        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for train_idx, val_idx in skf.split(selected_train_matrix,train_labels):
            tr_features,val_features = selected_train_matrix.iloc[train_idx],selected_train_matrix.iloc[val_idx]
            tr_labels,val_labels = train_labels.iloc[train_idx],train_labels.iloc[val_idx]

            model_cla = estimator
            model_cla.fit(tr_features.values, tr_labels.values)
            tra_cla_predictions = model_cla.predict(tr_features.values)
            val_cla_predictions = model_cla.predict(val_features.values)
            val_accuracy = accuracy_score(val_labels.values, val_cla_predictions)
            explainer = shap.TreeExplainer(model_cla)
            val_shap_values = explainer.shap_values(val_features.values)
            val_shap_expression = []
            for i in range(val_shap_values.shape[0]):
                expression = ""
                count = 0
                for j in range(len(selection)):
                    if selection[j]:

                        expression += f"Rule{j+1} SHAP value for high loading: {val_shap_values[i][count][1]}; "
                        count += 1
                val_shap_expression.append(expression) 
            
            train_indexes.extend(train_idx)
            valid_indexes.extend(val_idx)
            train_cla_predictions.extend(tra_cla_predictions)
            valid_cla_predictions.extend(val_cla_predictions)
            # all_val_performances.append(val_accuracy)
            shap_expressions_lst.extend(val_shap_expression)
        # all_val_performances = np.mean(np.array(all_val_performances), axis = 0)
        all_val_performances = accuracy_score(train_labels[valid_indexes].values,valid_cla_predictions)
        all_train_performances = accuracy_score(train_labels[train_indexes].values,train_cla_predictions)
        with open(f'{state.output_dir}/Iter_Accuracy_log.txt','a') as f:
            f.write(f'Validation Accuracy: {all_val_performances}; Train Accuracy: {all_train_performances}\n---------------------------------------------------\n')
        feedback = rules_feedback(train_set.iloc[valid_indexes], valid_cla_predictions, all_val_performances, shap_expressions_lst,tar_column_name)
    
        with open(f'{state.output_dir}/current_ML_metric.txt','w') as f1:
            f1.write(feedback)
        with open(f'{state.output_dir}/post_matrix_discussion.txt','a') as f:
            f.write(f'Current Accuracy and SHAP analysis:\n{feedback}')
        with open(f'{state.output_dir}/whole_log.txt','a') as f:
            f.write(f'ML Calculator Message:\n{feedback}\n---------------------------------------------------------------\n')
            
        #!save RFECV feature matrix (containing test)
        return {'messages':[BaseMessage(content=feedback,sender='ML Calculator')],'selected_train_matrix':selected_train_matrix_file,'selected_test_matrix':selected_test_matrix_file,'current_val_performance': all_val_performances}
    except ValueError:
        feedback = 'The Rule Matrix quality is awful. There is too much 0s which mean the rule hit few items in the Dataset SMILES. Please regenerate rules with higher quality. !!!!!*!!!!!'
        return {'messages':[BaseMessage(content=feedback,sender='ML Calculator')]}        
            
            
            
def ML_Commenter_o1(state:AgentState):
    '''
    Comparing to the reference accuracies and SHAP values and going through the accuracies and SHAP analysis, ML commenter gives comments and advice for further iteration of rules.
    '''
    #Give comment on accuracy and SHAP values
    GPT_model = state.GPT_model
    GPT_seed = state.GPT_seed
    # GPT_temperature = state.GPT_temperature
    with open(f'{state.output_dir}/current_rules.txt','r') as f:
        current_rules = f.read()
    with open(f'{state.output_dir}/ML_metric_log.txt','r') as f1:
        reference_ML_metrics = f1.read()
    current_ML_metric = state.messages[-1].content
    with open(f'{state.output_dir}/ML_metric_log.txt','a') as f1:
        f1.write(current_ML_metric)
    
    system_prompt = '''
    You are collaborating with other agents on a research program focused on a catalytic problem. 
    Please provide comments on the accuracies of the validation set and train set.
    Give analysis based on the SHAP value calcualted on the validation.
    Thinking about how to improve current rules which are used to generate feature matrix.
    '''
    user_prompt = f'''
    !!Reference Accuracis and SHAP vlaues!!
    {reference_ML_metrics}
    ---------------------------------------------------------------
    
    !!Current Accuracies and SHAP values!!
    {current_ML_metric}
    ----------------------------------------------------------------
    
    !!Current Rules!!
    {current_rules}
    ---------------------------------------------------------------
    
    !!Guidance for Analysis!!
    -1. Evaluate Effectiveness:
        -Are the current rules effective enough?
    -2. Assess Overfitting/Underfitting:
        -Are the iterative rules overfitting or underfitting to the current training and validation sets?
    -3. Analyze Discrepancies:
        -If the predicted loading is high while the experimental value is low for a molecule, the rule with the most positive SHAP value for high loading is likely problematic.
        -Conversely, if the predicted loading is low while the experimental value is high for a molecule, the rule with the most negative SHAP value for high loading is likely problematic.
    -4. Interpret Accuracy Improvements:
        -How do the accuracy metrics suggest the probability of improvement for the current rules?
    -5. Derive Insights from SHAP Analysis:
        -How does the SHAP analysis indicate opportunities for improving the current rules?
        -Note: You cannot directly use SHAP analysis results to modify the rules, but you should consider ways to improve the current rules based on these results.
    -----------------------------------------------------------------
    
    !!Performance Evaluation of Current Rules!!
    Please provide comments on the performance of current rules based on the latest training results. Use reference metric data as a baseline for your analysis. 
    The previous metrics should serve as a baseline reference, while the current iteration metrics should be used to determine if a local minimum has been reached and if there have been enough iterations to refine the rules.
    -----------------------------------------------------------------------------------------

    !!Recommendations for Improvement!!
    After your detailed analysis, please provide comments on how to improve the current rules. Consider the following points in your suggestions:
    Refine rules associated with high positive SHAP values for overestimated loadings.
    Adjust rules with high negative SHAP values for underestimated loadings.
    Improve generalization to address overfitting or underfitting issues.
    Use insights from SHAP analysis to iteratively refine and evaluate rule effectiveness.  
    '''
        
    combined_prompt = system_prompt + '\n----------------------------------------\n' + user_prompt    
        
    # while True:
    for i in range(100):
        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": combined_prompt}],
                model=GPT_model,
                temperature = 1,
                seed = GPT_seed,
                top_p = 1,
                n=1
            )
            output_message = chat_completion.choices[0].message.content.strip()
            with open(f'{state.output_dir}/whole_log.txt','a') as f:
                f.write(f'ML Commenter Message:\n{output_message}\n---------------------------------------------------------------\n')
                
            with open(f'{state.output_dir}/post_matrix_discussion.txt','a') as f:
                f.write(f'ML Commenter Message:\n{output_message}\n---------------------------------------------------------------\n')
                
            time.sleep(10)

            with open(f'{state.output_dir}/whole_log.txt','a') as f:
                f.write(f'Completion_tokens:\n{chat_completion.usage.completion_tokens}\nPrompt_tokens:\n{chat_completion.usage.prompt_tokens}---------------------------------------------------------------\n')    
            completion_tokens = state.completion_tokens + chat_completion.usage.completion_tokens
            prompt_tokens = state.prompt_tokens + chat_completion.usage.prompt_tokens

            return {'messages':[BaseMessage(content=output_message,sender='ML Commenter')],'completion_tokens':completion_tokens,'prompt_tokens':prompt_tokens}
        
        except TypeError or AttributeError:
            time.sleep(10)
            print('GPT RETURN ERROR, ML Commenter') 
