import os
import time

import numpy as np
import pandas as pd

from agent.state import AgentState,BaseMessage
from agent.client import client



# Rule metrics
def mean_support(matrix,labels):
    count = 0
    
    if len(matrix) != len(labels):raise ValueError("Matrix and Label are not equal in length")
    
    item_count = matrix.shape[0] * matrix.shape[1]
    for row_id in range(len(matrix)):
        label, row = labels[row_id], matrix[row_id]
        if label:
            for item in row:
                if item == 1:
                    count+=1
                    # print(row_id,item,label)
        else:
            for item in row:
                if item == -1:
                    count+=1
                    # print(row_id,item,label)
    if item_count == 0:return 0
    else:
        return count/item_count
def mean_confidence(matrix,labels):
    confidence = []
    rule_num = matrix.shape[1]
    sample_num = matrix.shape[0]
    #calculate confidence for each rule
    for column_id in range(rule_num):
        perfect = 0
        hit_count = 0
        column = matrix[:,column_id]
        for item_id in range(sample_num):
            item = column[item_id]
            label = labels[item_id]
            if item ==1 or item == -1:
                hit_count+=1 
                if item == 1 and label:
                    perfect+=1
                if item == -1 and not label:
                    perfect+=1
        if hit_count!=0:
            confidence.append(perfect/hit_count)
        else:confidence.append(0.5)
    # print(confidence)
    return np.mean(confidence)
def mean_lift(matrix,labels):
    lifts = []
    
    rule_num = matrix.shape[1]
    sample_num = matrix.shape[0]
    label_high_count = 0
    for label in labels:
        if label:label_high_count+=1
    prior_high = label_high_count/sample_num
    prior_low = 1 - prior_high
    #calculate confidence for each rule
    for column_id in range(rule_num):
        perfect_high = 0
        perfect_low = 0
        hit_count = 0
        column = matrix[:,column_id]
        for item_id in range(sample_num):
            item = column[item_id]
            label = labels[item_id]
            if item ==1 or item == -1:
                hit_count+=1 
                if item == 1 and label:
                    perfect_high+=1
                if item == -1 and not label:
                    perfect_low+=1
                              
        if hit_count!=0:
            high_lift = perfect_high/hit_count/prior_high
            low_lift = perfect_low/hit_count/prior_low  
            lifts.append(high_lift+low_lift)
        else:lifts.append(1)
    # print(lifts)
    
    return np.mean(lifts)
def mean_leverage(matrix,labels):
    leverage = []
    rule_num = matrix.shape[1]
    sample_num = matrix.shape[0]
    label_high_count = 0
    for label in labels:
        if label:label_high_count+=1
    prior_high = label_high_count/sample_num
    prior_low = 1 - prior_high
    #calculate confidence for each rule
    for column_id in range(rule_num):
        perfect_high = 0
        perfect_low = 0
        high_hit_count = 0
        low_hit_count = 0
        hit_count = 0
        column = matrix[:,column_id]
        for item_id in range(sample_num):
            item = column[item_id]
            label = labels[item_id]
            if item ==1 or item == -1:
                hit_count+=1
                if item == 1:high_hit_count+=1 
                if item == -1:low_hit_count+=1 
                if item == 1 and label:
                    perfect_high+=1
                if item == -1 and not label:
                    perfect_low+=1
                              
        if hit_count!=0:
            high_support = perfect_high/sample_num-high_hit_count/sample_num*prior_high
            low_support = perfect_low/sample_num-low_hit_count/sample_num*prior_low
            leverage.append(high_support+low_support)
        else:leverage.append(0)
    # print(supports)
    return np.mean(leverage)
    
def Metric_Calculator(state:AgentState):
    #Calculate the rule matrix
    train_features = np.array(pd.read_csv(os.path.join(state.output_dir,state.train_matrix)))
    train_set = pd.read_csv(os.path.join(state.output_dir,state.train_file))
    train_labels = train_set[f'{state.target_name}_high_or_low_value'].values
    train_support, train_confidence, train_lift, train_leverage = mean_support(train_features,train_labels), mean_confidence(train_features,train_labels),mean_lift(train_features,train_labels),mean_leverage(train_features,train_labels)
    output_message = f'''
    Train support: {train_support}
    Train confidence: {train_confidence}
    Train lift: {train_lift}
    Train leverage: {train_leverage}
    -------------------------------------------------------\n
    '''
    with open(f'{state.output_dir}/current_rule_metric.txt','w') as f1:
        f1.write(output_message)
    with open(f'{state.output_dir}/post_matrix_discussion.txt','a') as f:
        f.write(f'Current Rule Metrics:\n{output_message}')
    with open(f'{state.output_dir}/whole_log.txt','a') as f:
        f.write(f'Metric Calculator Message:\n{output_message}\n---------------------------------------------------------------\n')
    
    return {'messages':[BaseMessage(content=output_message,sender='Metric Calculator')],'current_mtx_gen':0}

def Metric_Commenter(state:AgentState):
    GPT_model = state.GPT_model
    GPT_seed = state.GPT_seed
    # GPT_temperature = state.GPT_temperature
    with open(f'{state.output_dir}/current_rules.txt','r') as f:
        current_rules = f.read()    
    with open(f'{state.output_dir}/rule_metric_log.txt','r') as f1:
        reference_metrics = f1.read()
    current_metric = state.messages[-1].content
    with open(f'{state.output_dir}/rule_metric_log.txt','a') as f1:
        f1.write(current_metric)
        
    system_prompt = '''
    You are collaborating with other agents on a research program focused on a catalytic problem. 
    Please provide comments on the performance of the current rules based on the current rule metrics.
    You should focus more on confidence and lift.
    '''
    user_prompt = f'''
    !!Explanation on rule metrics!!
    - **Support**: displays the proportion of records for which the entire rule, rule condition(s), and rule prediction(s), are true. For example, if 20% of the training data contains both bread and cheese, then rule support for the rule bread -> cheese is 20%.
    - **Support**: displays the ratio of rule support to rule condition support. This indicates the proportion of records with the specified rule condistion(s) for which the rule prediction(s) is/are also true. For example, if 50% of the training data contains bread (indicating rule condition support) but only 20% contains both bread and cheese (indicating rule support), then the prediction for the rule bread -> cheese would be Rule Support / Rule Condition Support or, in this case, 40%.
    - **Lift**: displays the ratio of confidence for the rule to the prior probability of having the rule prediction. For example, if 10% of the entire population buys bread, then a rule that predicts whether people will buy bread with 20% confidence will have a lift of 20/10 = 2. If another rule tells you that people will buy bread with 11% confidence, then the rule has a lift of close to 1, meaning that having the rule condition(s) does not make a lot of difference in the probability of having the rule prediction. In general, rules with lift different from 1 will be more interesting than rules with lift close to 1.
    - **Leverage**: displays the difference between the observed support for a rule and the expected support if the items were independent. Leverage measures the additional support the rule receives over what would be expected by chance. For example, if 5% of the training data contains both bread and cheese (indicating rule support), and if bread and cheese were independent with bread appearing in 20% and cheese in 10% of the data respectively, the expected support would be 20% * 10% = 2%. The leverage for the rule bread -> cheese would then be Rule Support - Expected Support or, in this case, 5% - 2% = 3%. Positive leverage values indicate a stronger association between the items than expected by chance, while negative leverage values indicate a weaker association. Leverage tends to highlight rules involving more frequent items in the dataset.
    -------------------------------------------------------------------
    
    !!Reference rule metrics!!
    {reference_metrics}
    ---------------------------------------------------------------
    
    !!Current Rules!!
    {current_rules}
    ---------------------------------------------------------------
    
    !!Current Metrics!!
    {current_metric}
    -------------------------------------------------------------------
    
    Please provide comments on the performance of the current rules based on the current metrics. You are encouraged to use reference metric data as a reference. 
    The previous metric should serve as a baseline reference. The metrics during the current iteration should be used to determine if a local minimum has been reached and if there have been enough iterations.
    Comment on how to improve the current rules after your  detailed analysis.
    '''
    
    # while True:
    for i in range(100):
        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                model=GPT_model,
                temperature = 0.3,
                seed = GPT_seed,
                top_p = 0.2,
                n=1
            )
            output_message = chat_completion.choices[0].message.content.strip()
            with open(f'{state.output_dir}/whole_log.txt','a') as f:
                f.write(f'Metric Commenter Message:\n{output_message}\n---------------------------------------------------------------\n')
            
            with open(f'{state.output_dir}/post_matrix_discussion.txt','a') as f:
                f.write(f'Metric Commenter Message:\n{output_message}\n---------------------------------------------------------------\n')
            
            time.sleep(10)

            return {'messages':[BaseMessage(content=output_message,sender='Metric Commenter')]}
        
        except TypeError or AttributeError:
            time.sleep(100)
            print('GPT RETURN ERROR, Metric Commenter')
            
            
            
            
            
def Metric_Commenter_o1(state:AgentState):
    GPT_model = state.GPT_model
    GPT_seed = state.GPT_seed
    # GPT_temperature = state.GPT_temperature
    with open(f'{state.output_dir}/current_rules.txt','r') as f:
        current_rules = f.read()    
    with open(f'{state.output_dir}/rule_metric_log.txt','r') as f1:
        reference_metrics = f1.read()
    current_metric = state.messages[-1].content
    with open(f'{state.output_dir}/rule_metric_log.txt','a') as f1:
        f1.write(current_metric)
        
    system_prompt = '''
    You are collaborating with other agents on a research program focused on a catalytic problem. 
    Please provide comments on the performance of the current rules based on the current rule metrics.
    You should focus more on confidence and lift.
    '''
    user_prompt = f'''
    !!Explanation on rule metrics!!
    - **Support**: displays the proportion of records for which the entire rule, rule condition(s), and rule prediction(s), are true. For example, if 20% of the training data contains both bread and cheese, then rule support for the rule bread -> cheese is 20%.
    - **Confidence**: displays the ratio of rule support to rule condition support. This indicates the proportion of records with the specified rule condistion(s) for which the rule prediction(s) is/are also true. For example, if 50% of the training data contains bread (indicating rule condition support) but only 20% contains both bread and cheese (indicating rule support), then the prediction for the rule bread -> cheese would be Rule Support / Rule Condition Support or, in this case, 40%.
    - **Lift**: displays the ratio of confidence for the rule to the prior probability of having the rule prediction. For example, if 10% of the entire population buys bread, then a rule that predicts whether people will buy bread with 20% confidence will have a lift of 20/10 = 2. If another rule tells you that people will buy bread with 11% confidence, then the rule has a lift of close to 1, meaning that having the rule condition(s) does not make a lot of difference in the probability of having the rule prediction. In general, rules with lift different from 1 will be more interesting than rules with lift close to 1.
    - **Leverage**: displays the difference between the observed support for a rule and the expected support if the items were independent. Leverage measures the additional support the rule receives over what would be expected by chance. For example, if 5% of the training data contains both bread and cheese (indicating rule support), and if bread and cheese were independent with bread appearing in 20% and cheese in 10% of the data respectively, the expected support would be 20% * 10% = 2%. The leverage for the rule bread -> cheese would then be Rule Support - Expected Support or, in this case, 5% - 2% = 3%. Positive leverage values indicate a stronger association between the items than expected by chance, while negative leverage values indicate a weaker association. Leverage tends to highlight rules involving more frequent items in the dataset.
    -------------------------------------------------------------------
    
    !!Reference rule metrics!! (mean metrics of previous rule matrices)
    {reference_metrics}
    ---------------------------------------------------------------
    
    !!Current Rules!!
    {current_rules}
    ---------------------------------------------------------------
    
    !!Current Metrics!!
    {current_metric}
    -------------------------------------------------------------------
    
    Please provide comments on the performance of the current rules based on the current metrics. You are encouraged to use reference metric data as a reference. 
    The previous metric should serve as a baseline reference. The metrics during the current iteration should be used to determine if a local minimum has been reached and if there have been enough iterations.
    Comment on how to improve the current rules after your  detailed analysis.
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
                f.write(f'Metric Commenter Message:\n{output_message}\n---------------------------------------------------------------\n')
            
            with open(f'{state.output_dir}/post_matrix_discussion.txt','a') as f:
                f.write(f'Metric Commenter Message:\n{output_message}\n---------------------------------------------------------------\n')
            
            time.sleep(10)

            with open(f'{state.output_dir}/whole_log.txt','a') as f:
                f.write(f'Completion_tokens:\n{chat_completion.usage.completion_tokens}\nPrompt_tokens:\n{chat_completion.usage.prompt_tokens}---------------------------------------------------------------\n')    
            completion_tokens = state.completion_tokens + chat_completion.usage.completion_tokens
            prompt_tokens = state.prompt_tokens + chat_completion.usage.prompt_tokens

            return {'messages':[BaseMessage(content=output_message,sender='Metric Commenter')],'completion_tokens':completion_tokens,'prompt_tokens':prompt_tokens}
        
        except TypeError or AttributeError:
            time.sleep(10)
            print('GPT RETURN ERROR, Metric Commenter')
            
            
            
            
def Metric_Commenter_o1_SMARTS(state:AgentState):
    GPT_model = state.GPT_model
    GPT_seed = state.GPT_seed
    # GPT_temperature = state.GPT_temperature
    with open(f'{state.output_dir}/current_rules.txt','r') as f:
        current_rules = f.read()    
    with open(f'{state.output_dir}/rule_metric_log.txt','r') as f1:
        reference_metrics = f1.read()
    current_metric = state.messages[-1].content
    with open(f'{state.output_dir}/rule_metric_log.txt','a') as f1:
        f1.write(current_metric)
        
    system_prompt = '''
    **System Prompt**
    
    You are collaborating with other agents on a research program focused on a catalytic problem. 
    Please provide comments on the performance of the current rules based on the current rule metrics.
    You should focus more on confidence and lift.
    
    ----
    '''
    user_prompt = f'''
    **Explanation on rule metrics**
    
    - **Support**: displays the proportion of records for which the entire rule, rule condition(s), and rule prediction(s), are true. For example, if 20% of the training data contains both bread and cheese, then rule support for the rule bread -> cheese is 20%.
    - **Support**: displays the ratio of rule support to rule condition support. This indicates the proportion of records with the specified rule condistion(s) for which the rule prediction(s) is/are also true. For example, if 50% of the training data contains bread (indicating rule condition support) but only 20% contains both bread and cheese (indicating rule support), then the prediction for the rule bread -> cheese would be Rule Support / Rule Condition Support or, in this case, 40%.
    - **Lift**: displays the ratio of confidence for the rule to the prior probability of having the rule prediction. For example, if 10% of the entire population buys bread, then a rule that predicts whether people will buy bread with 20% confidence will have a lift of 20/10 = 2. If another rule tells you that people will buy bread with 11% confidence, then the rule has a lift of close to 1, meaning that having the rule condition(s) does not make a lot of difference in the probability of having the rule prediction. In general, rules with lift different from 1 will be more interesting than rules with lift close to 1.
    - **Leverage**: displays the difference between the observed support for a rule and the expected support if the items were independent. Leverage measures the additional support the rule receives over what would be expected by chance. For example, if 5% of the training data contains both bread and cheese (indicating rule support), and if bread and cheese were independent with bread appearing in 20% and cheese in 10% of the data respectively, the expected support would be 20% * 10% = 2%. The leverage for the rule bread -> cheese would then be Rule Support - Expected Support or, in this case, 5% - 2% = 3%. Positive leverage values indicate a stronger association between the items than expected by chance, while negative leverage values indicate a weaker association. Leverage tends to highlight rules involving more frequent items in the dataset.
    
    ----
    
    **Reference rule metrics** (mean metrics of previous rule matrices)
    
    {reference_metrics}
    
    ----
    
    **Current Rules**
    
    {current_rules}
    
    ----
    
    **Current Metrics**
    
    {current_metric}
    
    ----
    
    Please provide comments on the performance of the current rules based on the current metrics. You are encouraged to use reference metric data as a reference. 
    The previous metric should serve as a baseline reference. The metrics during the current iteration should be used to determine if a local minimum has been reached and if there have been enough iterations.
    Comment on how to improve the current rules after your  detailed analysis.
    '''
    
    combined_prompt = system_prompt + user_prompt
    
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
                f.write(f'Metric Commenter Message:\n{output_message}\n---------------------------------------------------------------\n')
            
            with open(f'{state.output_dir}/post_matrix_discussion.txt','a') as f:
                f.write(f'Metric Commenter Message:\n{output_message}\n---------------------------------------------------------------\n')
            
            time.sleep(10)

            return {'messages':[BaseMessage(content=output_message,sender='Metric Commenter')]}
        
        except TypeError:
            time.sleep(10)
            print('GPT RETURN ERROR, Metric Commenter')