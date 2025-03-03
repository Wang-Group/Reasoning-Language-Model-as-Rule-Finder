import os
import time
import datetime

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from sklearn.metrics import accuracy_score

from agent.state import AgentState,BaseMessage,OverallState
from agent.client import client
from agent.rule_metric import mean_support,mean_confidence,mean_lift,mean_leverage


def read_last_lines_as_string(file_path, n=80):
    '''Read the last 80 lines in the rule discussion'''
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        last_lines = lines[-n:] if len(lines) >= n else lines
        return '\n'.join(last_lines)


def Final_Metric_loo(state:AgentState):
    '''
    Calculate all the metrics for the final rule sets.
    Returns:
        train_accuracy: accuracy on the train set.
        test_cla_predictions: classification prediction of the yield of the one test sample.
    '''

    seed = state.seed
    tar_column_name = f'{state.target_name}_high_or_low_value'
    target_name = state.target_name
    GPT_model = state.GPT_model
    train_set = pd.read_csv(os.path.join(state.output_dir,state.train_file))
    test_set = pd.read_csv(os.path.join(state.output_dir,state.test_file))
    train_features = pd.read_csv(os.path.join(state.output_dir,state.selected_train_matrix))
    train_labels = train_set[tar_column_name]
    test_features = pd.read_csv(os.path.join(state.output_dir,state.selected_test_matrix))
    test_labels = test_set[tar_column_name]
    
    if state.Tran_model == 'ETC':
        model_cla = ExtraTreesClassifier(n_estimators=500, random_state=seed,n_jobs=64)
    elif state.Tran_model == 'RFC':
        model_cla = RandomForestClassifier(n_estimators=500, max_depth = len(train_features.columns),random_state=seed,n_jobs=64)
    model_cla.fit(train_features.values, train_labels.values)
    train_cla_predictions = model_cla.predict(train_features.values)
    train_accuracy = accuracy_score(train_labels.values, train_cla_predictions) 
    test_cla_predictions = model_cla.predict(test_features.values)
    test_accuracy = accuracy_score(test_labels.values, test_cla_predictions)

    train_support, train_confidence, train_lift, train_leverage = mean_support(train_features.values,train_labels.values), mean_confidence(train_features.values,train_labels.values),mean_lift(train_features.values,train_labels.values),mean_leverage(train_features.values,train_labels.values)
    # test_support, test_confidence, test_lift, test_leverage = mean_support(test_features.values,test_labels.values), mean_confidence(test_features.values,test_labels.values),mean_lift(test_features.values,test_labels.values),mean_leverage(test_features.values,test_labels.values)
    test_support, test_confidence = mean_support(test_features.values,test_labels.values), mean_confidence(test_features.values,test_labels.values)
    with open(f'{state.output_dir}/current_rules.txt','r') as f:
        current_rules = f.read()
    with open(f'{state.output_dir}/{state.current_matrix}','r') as f:
        current_matrix = f.read()
 
    output_message1 = f'''
    Target Name: {target_name}
    GPT Model: {GPT_model}
    ML Model: {state.Tran_model}
    Current_Rules:
    {current_rules}\n
    Current_Matrix:
    {current_matrix}\n
    Train Accuracy: {train_accuracy}; Test Accuracy: {test_accuracy}
    Train Support: {train_support}; Test Support: {test_support}
    Train Confidence: {train_confidence}; Test Confidence: {test_confidence}
    Train Lift: {train_lift}; Test Lift: NONE
    Train Leverage: {train_leverage}; Test Leverage: NONE
    ----------------------------------------------------------------------------------\n
    '''   
    with open(f'{state.output_dir}/final_output.txt','a') as f:
        f.write(output_message1)
    if os.path.exists(f'{state.output_dir}/current_rule_code.txt'):
        with open(f'{state.output_dir}/current_rule_code.txt','r') as f:
            current_rule_code = f.read()
        with open(f'{state.output_dir}/final_output.txt','a') as f:
            f.write('Current Rule Code:\n')
            f.write(current_rule_code+'\n')
            f.write('-'*50)
    with open(f'{state.output_dir}/whole_log.txt','a') as f:
        f.write(f'Final Metric Message:\n{output_message1}\n---------------------------------------------------------------\n')

    return train_accuracy,test_cla_predictions
            
def Manager_loo_o1(state:AgentState):
    '''
    Based on all the metrics and the discussion after matrix generation, decide whether to end the loop (a loop containing all the processing for one train-test split, i.e., several iterations).
    Give suggestions for continuous rule iteration if decide to continue.
    '''
    GPT_model = state.GPT_model
    GPT_seed = state.GPT_seed
    Fe_pred_flag = state.Fe_pred_flag
    
    train_path = os.path.join(state.output_dir,state.train_file)
    train_set = pd.read_csv(train_path).iloc[:,1:]#read name, SMILES and Classification target
    reaction_background = state.reaction_background
    if Fe_pred_flag:
        Fe_loading_pred_train = pd.read_csv(os.path.join(state.output_dir,'Fe_pred_train.csv'))
        reaction_background += '\n Note that there are predictions given by Fe loading data with a considerable precision. Your rules should be some of additional help.'
        train_set = pd.concat([train_set,Fe_loading_pred_train],axis=1)
    with open(f'{state.output_dir}/current_rules.txt','r') as f:
        current_rules = f.read()
    with open(f'{state.output_dir}/current_rule_metric.txt','r') as f:
        current_rule_metric = f.read()
    with open(f'{state.output_dir}/rule_metric_log.txt','r') as f1:
        rule_metric_log = f1.read()
    with open(f'{state.output_dir}/current_ML_metric.txt','r') as f1:
        current_ML_metric = f1.read()
    with open(f'{state.output_dir}/ML_metric_log.txt','r') as f1:
        ML_metric_log = f1.read()

    rule_discussions = ''
    discussion_role = ['Matrix Checker','Metric Commenter','ML Commenter']
    for message in state.messages[-6:]: 
        if message.sender in discussion_role:
            rule_discussions = rule_discussions + '\n' + message.content
        
    system_prompt = '''
    You are the manager of a target aimed at extracting rules from SMILES to describe the catalytic action of a modified catalyst.
    Please give your suggestions to improve current rules.
    '''    
    user_prompt = f'''
    !! Reaction Background !!
    {reaction_background}
    ----------------------------------------------------------------------
    
    !! Rule Metric Log !!
    {rule_metric_log}
    ----------------------------------------------------------------------
    
    !! Accuracy and SHAP Log !!
    {ML_metric_log}
    ----------------------------------------------------------------------
    
    !! Training  Set !!
    {train_set}
    ----------------------------------------------------------------------
    
    !! Current Rules !!
    {current_rules}
    ---------------------------------------------------------------------
    
    !! Current Rule Metric !!
    {current_rule_metric}
    ---------------------------------------------------------------------
    
    !! Current Accuracy and SHAP !!
    {current_ML_metric}
    ----------------------------------------------------------------------
    
    !! Rule Discussions !!
    {rule_discussions}
    ---------------------------------------------------------------------
    
    !! Your Target !!
    1. Summarize the discussions above and provide directions to optimize the current rules. Please give your suggestions to improve current rules.
    2. Judge if there is any possibility for further optimization of the current rules without overfitting. 
    3. If you think the rules should be further optimized, write '**Please Optimize Rules**' at the end of your answer. If you think there is enough optimization, do not add anything and let the program shutdown.
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
                f.write(f'Project Manager Message:\n{output_message}\n---------------------------------------------------------------\n')
                
            with open(f'{state.output_dir}/post_matrix_discussion.txt','a') as f:
                f.write(f'Project Manager Message:\n{output_message}\n---------------------------------------------------------------\n')
                
            train_accuracy,test_prediction = Final_Metric_loo(state)
            train_accuracies = state.train_performance
            test_predictions = state.test_prediction

            test_predictions.append(bool(test_prediction))
            train_accuracies.append(state.current_val_performance) # read the 5 cv validation accuracy on the train set

            time.sleep(10)
            
            with open(f'{state.output_dir}/whole_log.txt','a') as f:
                f.write(f'Completion_tokens:\n{chat_completion.usage.completion_tokens}\nPrompt_tokens:\n{chat_completion.usage.prompt_tokens}---------------------------------------------------------------\n')    
            completion_tokens = state.completion_tokens + chat_completion.usage.completion_tokens
            prompt_tokens = state.prompt_tokens + chat_completion.usage.prompt_tokens
            
            now = datetime.datetime.now()
            with open(f'{state.output_dir}/whole_log.txt','a') as f:
                f.write(f'\n---------------------------------------------------------------\nEND Time: {str(now)}\n---------------------------------------------------------------\n')
            return {'messages':[BaseMessage(content=output_message,sender='Project Manager')],'train_performance':train_accuracies,"test_prediction": test_predictions,'completion_tokens':completion_tokens,'prompt_tokens':prompt_tokens}
        
        except TypeError or AttributeError as e:
            time.sleep(10)
            print('GPT RETURN ERROR, Project Manager')
            print(e)
            