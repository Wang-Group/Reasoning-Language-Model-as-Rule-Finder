import pandas as pd
import os
from langgraph.constants import Send

from agent.state import AgentState,OverallState,BaseMessage



#edge_utilities
def good_rules_advisor(state:AgentState):
    '''Positioned after the rule advisor, it determines whether to forward to the matrix generator node based on the adequacy of the rules or if they have been rejected three times.'''
    if state.messages[-2].sender == 'Matrix Generator':
        return 'rule generator'
    else:
        current_gen_count = state.current_gen_count
        if current_gen_count < 2:# change to two for cost
            if "**TRUE**" in state.messages[-1].content:
                return 'matrix generator'
            else:
                return 'rule generator'
        else:return 'matrix generator'

def if_comment(state:AgentState):
    '''If two many generations, do not comment. current_gen_count will refresh after matrix generated successfully.'''
    current_gen_count = state.current_gen_count
    if current_gen_count < 2:# change to two for cost
        return 'rule commenter'
    else:
        return 'matrix generator'
    
def good_rules_commenter(state:AgentState):
    '''Checke the comment of Rule Commenter: go to matrix generator node if the rules are good enough'''
    if "**TRUE**" in state.messages[-1].content:
        return 'matrix generator'
    else:
        return 'rule advisor'

# def too_many_generations(state:AgentState):
#     '''END the process if to many times are tried for rule generation'''
#     if state.generate_count > 30:
#         print('END because of too many generations')
#         return '__end__'
#     else:
#         return 'rule commenter'

def matrix_generate_error(state:AgentState):
    '''If it's hard to generate matrix, regenerate the rule'''
    if '**Matrix Generate Error**' in state.messages[-1].content:
        return 'rule advisor'
    else:
        return 'matrix checker'

def judge_matrix_checker(state:AgentState):
    '''Determine to go to next process if rule2matrix is effective'''
    if state.current_mtx_gen < 2:# change to two times for cost
        if "**TRUE**" in state.messages[-1].content:
            return 'metric calculator'
        else:
            return 'matrix generator'
    else:return 'metric calculator'

# def judge_manager(state:AgentState):
#     '''Determine to END the loop if the existing result is good enough'''
#     if "**Please Optimize Rules**" in state.messages[-1].content:
#         return 'rule advisor'
#     elif state.generate_count > 50:
#         return '__end__'
#     else:
#         return '__end__'
    
def tradition_calc_exception(state:AgentState):
    '''exception of tree models may be met when rule matrix has too much 0s'''
    if '!!!!!*!!!!!' in state.messages[-1].content:
        return 'rule advisor'
    else:return "traditional commenter"
    
# def check_performance(train_performances: list):
    
#     '''
#     Check the latest 5 performance and decide whether to shut the loop
#     '''

#      # If fewer than 5 performances, return the index of best performance immediately
#     if len(train_performances) <= 6:
#         best_index = 0
#         best_value = float(0)
#         for i in range(len(train_performances)):
#             if train_performances[i] >= best_value:
#                 best_index = i
#                 best_value = train_performances[i]
        
#         return 'Go On',best_index
     
#     else:
#         # Get the latest 5 performances
#         recent_performances = train_performances[-5:]
        
#         #record latest best performance
#         best_index = 0
#         best_value = float(0)
#         for i in range(len(train_performances)-5):
#             if train_performances[i] >= best_value:
#                 best_index = i + 5
#                 best_value = train_performances[best_index]
    
#         # If there is no improvement in the latest 5 performances, return the index of the best performance
#         if all(perf <= train_performances[best_index] for perf in recent_performances[:-1]):
#             return 'Shut Down',best_index
#         # Otherwise, return the index of the latest best performance within the last 5 updates
#         return 'Go On',best_index

def check_performance(train_performances: list):
    
    '''
    Check the latest 5 performance(accuracy) and shut the loop if there is no improvement.
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
    '''Determine to END the loop if the existing result is good enough or no improvement is in the latest five iteration'''
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
    # elif state.generate_count > 100:
    #     return "one output"
    else:
        return "one output"
def split_1_out(state:AgentState):
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
    
# def continue_LOO(state:OverallState):
#     #send trainset,test_set,folder
#     output_dir = state['output_folder']
#     GPT_model = state['GPT_model']
#     target_name = state['target_name']
    
#     return [Send("one splitting", AgentState(messages=[BaseMessage(content=f'Data Load Successfully for {target_name}',sender='load_data')],GPT_model=GPT_model,target_name=target_name,output_dir=output_dir+'/'+str(s),generate_count=0)) for s in range(1,37)]

def cv_out(state:AgentState):
    train_performances = state.train_performance
    test_preds = state.test_prediction
    # best_index = check_performance(train_performances)
    best_index = 0
    best_value = float(0)
    for i in range(len(train_performances)): #latest best value
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
    current_output = 'Iteration:' + iter_num + '\n' + f'Best Train Accuracy: {train_performance}; Corresponding Test True:{list(test_true)},Test Pred: {test_pred}\n' + '-'*50 + '\n'
    with open(os.path.join(output_dir,'..','loo_out_log.txt'),'a') as f:
        f.write(current_output)
    # return {'loo_out':[{'Best train_performance':train_performance,'test_true':test_true,'test_pred':test_pred}]}
    return {'messages':[BaseMessage(content=current_output,sender='one split out')]}
    