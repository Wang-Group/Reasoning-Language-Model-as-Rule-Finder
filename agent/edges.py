import pandas as pd
import os
from langgraph.constants import Send

from agent.state import AgentState, BaseMessage



# edge_utilities for langgraph
def good_rules_advisor(state:AgentState):
    '''
    Positioned after the rule advisor, it determines whether to forward to the matrix generator node based on the adequacy of the rules or if they have been rejected three times.
    Return:
        the next node to go to. 
        If Rule Advisor is satisfied with current rules, or the number of rule generation is larger than 1, or the request of rule regeneration is not for the code error in Matrix Generator, the next node is Matrix Generator.
        Or the next node is Rule Generator.
    '''
    if state.messages[-2].sender == 'Matrix Generator':
        return 'rule generator'
    else:
        current_gen_count = state.current_gen_count
        if current_gen_count < 3:# change to two for cost
            if "**TRUE**" in state.messages[-1].content:
                return 'matrix generator'
            else:
                return 'rule generator'
        else:return 'matrix generator'

def if_comment(state:AgentState):
    '''
    If two many generations, do not comment. current_gen_count will refresh after matrix generated successfully.
    Return:
        the next node to go to.
        If the number of rule generation is smaller than 2, the next node is Rule Commenter.
        Or the next node is Matrix Generator
    '''
    current_gen_count = state.current_gen_count
    if current_gen_count < 3:# change to two for cost
        return 'rule commenter'
    else:
        return 'matrix generator'
    
def good_rules_commenter(state:AgentState):
    '''
    Checke the comment of Rule Commenter: go to matrix generator node if the rules are good enough.
    Return:
        the next node to go to.
        If Rule Commenter is satisfied with the rules, the next node is Matrix Generator.
        Or the next node is Rule Advisor.
    '''
    if "**TRUE**" in state.messages[-1].content:
        return 'matrix generator'
    else:
        return 'rule advisor'

def matrix_generate_error(state:AgentState):
    '''
    If an error is encountered when running code of matrix generation, regenerate the rules.
    Return:
        the next node to go to.
        If there is any error when running the code for matrix generation, the next node is Rule Advisor.
        Or the next node is Matrix Checker.
    '''
    if '**Matrix Generate Error**' in state.messages[-1].content:
        return 'rule advisor'
    else:
        return 'matrix checker'

def judge_matrix_checker(state:AgentState):
    '''
    Determine to go to next process if rule2matrix is effective.
    Return:
        the next node to go to. 
        If Matrix Checker is unsatisfied with current feature matrix, or the number of matrix generation is smaller than 2, the next node is Matrix Generator.
        Or the next node is Metric Calculator.
    '''
    if state.current_mtx_gen < 3:# change to two times for cost
        if "**TRUE**" in state.messages[-1].content:
            return 'metric calculator'
        else:
            return 'matrix generator'
    else:return 'metric calculator'


def ML_calc_exception(state:AgentState):
    '''
    Handle the exception of tree models when rule matrix has too much 0s.
    Return:
        the next node to go to.
        If there is any error when calculating ML metrics (accuracy and SHAP values), the next node is Rule Advisor.
        Or the next node is ML Commenter.
        '''
    if '!!!!!*!!!!!' in state.messages[-1].content:
        return 'rule advisor'
    else:return "ML commenter"

def check_performance(train_performances: list):
    '''
    Check if there is any improvement in the latest 5 accuracies. If not, end the loop (a loop containing all the processing for one train-test split, i.e., several iterations) for this split.
    Args:
        train_performance: list.
            The recorded accuracies for this split.
    Returns:
        Flag of whether to shut down the loop of this iteration
        best_index: 
            The index of highest accuracy in train_performance. If there is not only 1 high accuracy, the latest one is chosen.
    '''
    
    best_index = 0
    best_value = float(0)
    for i in range(len(train_performances)):
        if train_performances[i] >= best_value:
            best_index = i
            best_value = train_performances[i]
            
     # If fewer than 6 performances, return the index of best performance immediately
    if len(train_performances) <= 6:
        return 'Go On',best_index
     
    else:
        # Check if there is any improvement in the latest five accuracies
        best_index_improved = 0
        best_value_improved = float(0)
        for i in range(len(train_performances)):
            if train_performances[i] > best_value_improved:
                best_index_improved = i
                best_value_improved = train_performances[i]
                
        if best_index_improved < len(train_performances) - 5:
            return 'Shut Down',best_index
        
        # Otherwise, return the index of the latest best performance within the last 5 updates
        return 'Go On',best_index

def judge_manager_loo(state:AgentState):
    '''
    Determine to END the loop (a loop containing all the processing for one train-test split, i.e., several iterations) if the existing result is good enough or no improvement is in the latest 5 iteration.
    Return:
        the next node to go to.
        If Project Manager is satisfied with current rules and feature matrix, or there is no improvement in the latest 5 iterations, the next node is one output. This will end the loop of this split.
        Or the next node is Rule Advisor. This will start a new iteration (From Rule Generator to Project Manager). 
    '''
    if "Please Optimize Rules" in state.messages[-1].content:
        # if no update in the latest 5 train performances, stop iteration
        train_performances = state.train_performance
        flag,best_index = check_performance(train_performances)
        
        if best_index == len(train_performances) - 1:
            train_matrix = pd.read_csv(os.path.join(state.output_dir,state.train_matrix))
            test_matrix = pd.read_csv(os.path.join(state.output_dir,state.test_matrix))
            selected_train_matrix = pd.read_csv(os.path.join(state.output_dir,state.selected_train_matrix))
            selected_test_matrix = pd.read_csv(os.path.join(state.output_dir,state.selected_test_matrix))
            train_matrix.to_csv(os.path.join(state.output_dir,'best_train_matrix.csv'),index=None)
            test_matrix.to_csv(os.path.join(state.output_dir,'best_test_matrix.csv'),index=None)
            selected_train_matrix.to_csv(os.path.join(state.output_dir,'best_selected_train_mtx.csv'),index=None)
            selected_test_matrix.to_csv(os.path.join(state.output_dir,'best_selected_test_mtx.csv'),index=None)
            if os.path.exists(os.path.join(state.output_dir,'current_rule_code.txt')):
                with open(os.path.join(state.output_dir,'current_rule_code.txt'),'r') as f:
                    code = f.read()
                with open(os.path.join(state.output_dir,'best_rule_code.txt'),'w') as f1:
                    f1.write(code)
            with open(os.path.join(state.output_dir,'current_rules.txt'),'r') as f:
                code = f.read()
            with open(os.path.join(state.output_dir,'best_rules.txt'),'w') as f1:
                f1.write(code)
                
        else:pass
        
        if flag == 'Go On':
            return 'rule advisor'
        else:
            return "one output"
    else:
        return "one output"
    
def split_1_out(state:AgentState):
    '''
    A node ending the loop (a loop containing all the processing for one train-test split, i.e., several iterations) and logging the metrics on the train set and prediction on the test set. 
    '''
    train_performances = state.train_performance
    test_preds = state.test_prediction
    # best_index = check_performance(train_performances)
    #Find the index of the latest highest performance
    best_index = 0
    best_value = float(0)
    for i in range(len(train_performances)):
        if train_performances[i] >= best_value:
            best_index = i
            best_value = train_performances[i]
           
    train_performance = train_performances[best_index]
    test_pred = test_preds[best_index]
    tar_column_name = f'{state.target_name}_high_or_low_value'
    test_set = pd.read_csv(os.path.join(state.output_dir,state.test_file))
    test_true = test_set[tar_column_name]
    test_true = test_true.values
    output_dir = state.output_dir
    iter_num = os.path.basename(output_dir)
    current_output = 'Iteration:' + iter_num + '\n' + f'Best Train Accuracy: {train_performance}; Corresponding Test True:{bool(test_true)},Test Pred: {bool(test_pred)}\n' + '-'*50 + '\n'
    
    with open(f'{state.output_dir}/whole_log.txt','a') as f:
        f.write(f'Total Completion_tokens:\n{state.completion_tokens}\nTotal Prompt_tokens:\n{state.prompt_tokens}---------------------------------------------------------------\n')    

    with open(os.path.join(output_dir,'..','loo_out_log.txt'),'a') as f:
        f.write(current_output)
        
    # return {'loo_out':[{'Best train_performance':train_performance,'test_true':test_true,'test_pred':test_pred}]}
    return {'messages':[BaseMessage(content=current_output,sender='one split out')]}
    
    