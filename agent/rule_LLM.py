import os
import time
import datetime

import numpy as np
import pandas as pd
import pkg_resources

from agent.state import AgentState,BaseMessage
from agent.client import client

def read_last_lines_as_string(file_path, n=80):
    '''
    Read the last 80 lines in the rule discussion
    Args:
        file_path: the location of the logged rule discussion.
        n: last n lines to read. For the token limit of LLMs. 
    Return:
        The content of the last n lines in the file.
    '''
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        last_lines = lines[-n:] if len(lines) >= n else lines
        return '\n'.join(last_lines)
            
def Rule_Generator_o1(state:AgentState):
    '''Given the dataset with name, SMILES, and yield, together with background information on the catalytic reaction, key chemistry, and other existing advice, the Rule Generator try to propose rules describing the relationship between the structures of the modifiers and the reaction yield (high/low).'''
    now = datetime.datetime.now()
    with open(f'{state.output_dir}/whole_log.txt','a') as f:
        f.write(f'\n---------------------------------------------------------------\nSTART Time: {str(now)}\n---------------------------------------------------------------\n')
    current_gen_count = state.current_gen_count
    generate_count = state.generate_count
    GPT_model = state.GPT_model
    GPT_seed = state.GPT_seed
    Fe_pred_flag = state.Fe_pred_flag
    # GPT_temperature = state.GPT_temperature
    train_path = os.path.join(state.output_dir,state.train_file)
    train_set = pd.read_csv(train_path).iloc[:,1:]#skip original regression target value column
    if os.path.exists(f'{state.output_dir}/current_rules.txt'):
        with open(f'{state.output_dir}/current_rules.txt','r') as f:
            current_rules = f.read()
        current_rules = '\n !!Current Rules!!\n There may be advise based on imporvement of current rules, here are current rules:\n'+current_rules
    else:
        current_rules = ''
    reaction_background = '''
    You are tasked with solving the following problem: a radical-mediated remote δ-C(sp3)–H bond functionalization reaction of aliphatic alcohols using di-tert-butyl azodicarboxylate (DBAD) as the substrate. The reaction is catalyzed by FeCl3 in the presence of tetrabutylammonium chloride (TBACl) and conducted in acetonitrile solvent under irradiation with 390 nm light-emitting diodes (LEDs).

    In addition, the reaction employs Hf-TPY-MOL, a Metal Organic Layer composed of hafnium-oxygen clusters (SBU, Secondary Building Unit) coordinated with terpyridine ligands. This setup is used to capture and stabilize the Fe ion. The SBU of the MOL can be modified using a molecular modifier to affect the reactivity of the catalyst Hf-TPY-MOL(Fe).

    The primary goal is to optimize and control the yield of the remote δ-C(sp3)–H bond functionalization reaction. It has been observed that the modifier loading on the catalyst (modifier/SBU), the fraction of Fe to Hf in the catalyst (Fe/Hf), and the total loading of Fe (Fe_loading) significantly impact the yield. It's assumed that modifier/SBU, Fe/Hf, and yield These parameters are influenced by different types of molecular modifiers.'''
    
    if Fe_pred_flag:
        Fe_loading_pred_train = pd.read_csv(os.path.join(state.output_dir,'Fe_pred_train.csv'))
        reaction_background += '\n Note that there are predictions given by Fe loading data with a considerable precision. Your rules should be some of additional help.'
        train_set = pd.concat([train_set,Fe_loading_pred_train],axis=1)
        
    system_prompt = f'''
    You are collaborating with other agents on a research program focused on the aforementioned problem. Utilizing your chemical insights, please analyze the dataset provided below and generate rules that describe the relationship between molecular modifiers (depicted as SMILES strings) and the relative high or low value of {state.target_name}.
    
    If any advice is given, take the advice into consideration.
    '''
    user_prompt = f'''
    !!Reaction Background!!
    {reaction_background}
    --------------------------------------------------------------------------------------------------------------------
    
    !!DataSet!!
    {train_set}
    --------------------------------------------------------------------------------------------------------------------
    
    !! Requirements for Generated Rules !!
    1. Rules should illustrate direct combinations of sub-structures (function groups) features in the modifiers. Combining multiple sub-structures (function groups) is recommended.
    2. When generating rules, consider the underlying physical-chemical properties in your mind.
    3. Each rule should clearly predict whether the target value is high or low for any SMILES structure that fits its description.
    4. Prioritize rules that cover a broader range of the dataset.
    5. Generate between 5 and 15 rules.
    6. Maintain a suitable balance between simple rules with higher coverage and complex rules with lower coverage.
    7. If you think some of the current rules are terrible, abandon them and write new ones.
    ----------------------------------------------------------------------------------------------------------------------
    
    !!Format of Rules!!
    **Start of Rules
    - **Rule 1**:...
    - **Rule 2**:...
    - ...
    **End of Rules
    -----------------------------------------------------------------------------------------------------------------------\n
    
    {current_rules}
    '''
    add_prompt = '''\n-------------------------------------------------------------------------------------------------------\nPlease generate rules! You're suggested to polish the current rule set'''
    
    rule_advice = ''
    last_msg = state.messages[-1].content
    if last_msg.find("** Start of Advice **") != -1 and state.messages[-1].sender == 'Rule Advisor':
        rule_advice = state.messages[-1].content

    if rule_advice != '':
        add_prompt = rule_advice+ add_prompt
    
    combined_prompt = system_prompt + '\n----------------------------------------\n' + user_prompt
        
    # while True:
    for i in range(100):
        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": combined_prompt}],
                model=GPT_model,
                temperature = 1,
                seed = GPT_seed,
                top_p=1,
                n=1
            )
            
            output_message = chat_completion.choices[0].message.content.strip()
            with open(f'{state.output_dir}/whole_log.txt','a') as f:
                f.write(f'Rule Generator Message:\n{output_message}\n---------------------------------------------------------------\n')
            with open(f'{state.output_dir}/Rule_discussion_log.txt','a') as f:
                f.write(f'Rule Generator Message:\n{output_message}\n---------------------------------------------------------------\n')    
            with open(f'{state.output_dir}/current_rules.txt','w') as f:
                f.write(f'{output_message}---------------------------------------------------------------\n')
                
                
           
            # output_message  = extract_rules(output_message)
            generate_count += 1
            current_gen_count += 1
            time.sleep(10)
            
            with open(f'{state.output_dir}/whole_log.txt','a') as f:
                f.write(f'Completion_tokens:\n{chat_completion.usage.completion_tokens}\nPrompt_tokens:\n{chat_completion.usage.prompt_tokens}---------------------------------------------------------------\n')    
            completion_tokens = state.completion_tokens + chat_completion.usage.completion_tokens
            prompt_tokens = state.prompt_tokens + chat_completion.usage.prompt_tokens
            
            return {'messages':[BaseMessage(content=output_message,sender='Rule Generator')],'reaction_background':reaction_background,'generate_count':generate_count,'current_gen_count':current_gen_count,'completion_tokens':completion_tokens,'prompt_tokens':prompt_tokens}
        
        except TypeError:
            time.sleep(10)
            print('GPT RETURN ERROR, Rule Generator')
    
    
def Rule_Commenter_o1(state:AgentState): 
    '''Rule Commenter gives commments on current rules based on clarity, property insight, complexity, coverage, and balance. If he is satisfied with the current rules, ''' 
    GPT_model = state.GPT_model
    GPT_seed = state.GPT_seed
    # GPT_temperature = state.GPT_temperature
    train_path = os.path.join(state.output_dir,state.train_file)
    train_set = pd.read_csv(train_path).iloc[:,1:]#skip original regression target value column
    current_rules = state.messages[-1].content
    reaction_background = state.reaction_background
    Fe_pred_flag = state.Fe_pred_flag
    
    if Fe_pred_flag:
        Fe_loading_pred_train = pd.read_csv(os.path.join(state.output_dir,'Fe_pred_train.csv'))
        train_set = pd.concat([train_set,Fe_loading_pred_train],axis=1)
        
        
    system_prompt = '''
    You are collaborating with other agents on a research program focused on the a catalytical problem. 
    Your target is give comment to rules for advance rules and judge if the rules is enough for latter tasks.
    '''
    user_prompt=f'''
    !!Reaction Background!!
    {reaction_background}
    
    -------------------------------------------------------------------------------------------------
    !!Current Rules!!
    {current_rules}
    ------------------------------------------------------------------------------------------------
    
    !!DataSet!!
    {train_set}
    ------------------------------------------------------------------------------------------------
    
    !! Scoring Criteria for Rules !!
    # Importance in descending order:
    1. **Clarity**: Determine if you can clearly tell whether the target value is high or low when a modifier matches the structural description of the rule.
    2. **Property Insight**: Assess whether there is adequate physical-chemical insight corresponding to the properties of the modifier and the reaction.
    3. **Complexity**: Ensure some rules consider combinations of sub-structures (functional groups) rather than a single functional group.
    4. **Coverage**: Verify that at least 2 data points support a rule. More support corresponds to a higher score.
    5. **Balance**: Consider the balance between complexity and coverage. Although complex rules with detailed sub-structures are valuable, sometimes simpler rules with higher coverage are also effective. Try to achieve a balanced approach.
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    
    !!Comment Format!!
    **Start of Comments
    - **Comment 1**:...
    - **Comment 2**:...
    - ...
    **End of Comments
    ----------------------------------------------------------------------------------------------------------------------------------------------------------
    
    !!Your Target!!
    Scan the rules one by one and score them based on the four criteria above. Sum the scores for each one rule and provide your comments on the rules.
    If you believe the rules are good enough, conclude your answer with "**TRUE**", if not, DO NOT add anything in the end.
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
                f.write(f'Rule Commenter Message:\n{output_message}\n---------------------------------------------------------------\n')
            with open(f'{state.output_dir}/Rule_discussion_log.txt','a') as f:
                f.write(f'Rule Commenter Message:\n{output_message}\n---------------------------------------------------------------\n')  
            
            time.sleep(10)
            
            with open(f'{state.output_dir}/whole_log.txt','a') as f:
                f.write(f'Completion_tokens:\n{chat_completion.usage.completion_tokens}\nPrompt_tokens:\n{chat_completion.usage.prompt_tokens}---------------------------------------------------------------\n')    
            completion_tokens = state.completion_tokens + chat_completion.usage.completion_tokens
            prompt_tokens = state.prompt_tokens + chat_completion.usage.prompt_tokens
            return {'messages':[BaseMessage(content=output_message,sender='Rule Commenter')],'completion_tokens':completion_tokens,'prompt_tokens':prompt_tokens}
        
        except TypeError:
            time.sleep(10)
            print('GPT RETURN ERROR, Rule Commenter')
    
def Rule_Advisor_o1(state:AgentState):
    '''
    Considering the situation, the Rule Advisor will give advice for updating rules.
    There are 3 situations for regenerating rules:
        1. Rule Commenter is unsatisfied with the newly generated rules.
        2. Matrix Generator encounters an error when running the generated code.
        3. Project Manager gives suggestions for generating new rules, which starts a new iteration (From Rule Generator to Project Manager). 
    '''
    GPT_model = state.GPT_model
    GPT_seed = state.GPT_seed
    
    Fe_pred_flag = state.Fe_pred_flag
    
    # GPT_temperature = state.GPT_temperature
    train_path = os.path.join(state.output_dir,state.train_file)
    train_set = pd.read_csv(train_path).iloc[:,1:]#skip original regression target value column
    
    if Fe_pred_flag:
        Fe_loading_pred_train = pd.read_csv(os.path.join(state.output_dir,'Fe_pred_train.csv'))
        train_set = pd.concat([train_set,Fe_loading_pred_train],axis=1)

    with open(f'{state.output_dir}/current_rules.txt','r') as f:
        current_rules = f.read()
    reaction_background = state.reaction_background
    # with open(f'{state.output_dir}/Rule_discussion_log.txt','a') as f:
    #     input_discussion = f.read()
    if state.messages[-1].sender == 'Rule Commenter':
        # with open(f'{state.output_dir}/Rule_discussion_log.txt','r') as f:
        #    input_discussion = f.read()
        file_path = f'{state.output_dir}/Rule_discussion_log.txt'
        input_discussion = read_last_lines_as_string(file_path, n=500)
           
    elif state.messages[-1].sender == 'Project Manager':
        # with open(f'{state.output_dir}/post_matrix_discussion.txt','r') as f:
        #     input_discussion = f.read()
        input_discussion = state.messages[-1].content
        input_discussion = 'Project Manager Message:\n'+ input_discussion
        input_discussion = input_discussion + "!!Please improve current rules based on the discussion!!"
    elif state.messages[-1].sender == 'Matrix Generator':
        input_discussion = '''It's difficult to generate numeric feature matrix from current rules.\n''' + state.messages[-1].content
        # input_discussion = state.messages[-1].content
    elif state.messages[-1].sender == 'ML Calculator':
        # Model training error because of almost all 0 matrix. This error will not be encountered by o1, for its effective code for rules
        input_discussion = state.messages[-1].content
    system_prompt='''
    You are tasked with reading the discussion on the current rules. 
    Based on the reaction data, provide practical advice for the rule generator to create improved rules. 
    '''
    user_prompt=f'''
    !!Reaction Background!!
    {reaction_background}
    -------------------------------------------------------------------------------------------------
    
    !!Current Rules!!
    {current_rules}
    ------------------------------------------------------------------------------------------------
    
    !!DataSet!!
    {train_set}
    ------------------------------------------------------------------------------------------------
    
    !!Discussions!!
    {input_discussion}
    ------------------------------------------------------------------------------------------------
    
   !! Advice Format !!
    ** Start of Advice **
    - ** Advice 1 **: ...
    - ** Advice 2 **: ...
    - ...
    ** End of Advice **

    
    -------------------------------------------------------------------------------------------------
    
    !!Your Target!!
    Your advice must be directly practical on how to improve the SMILES based rules set or directly give new rules. 
    If you believe the rules are good enough in this stage, conclude your answer with "**TRUE**". If not, DO NOT add anything at the end.
    Note that advice from Project Manager is important, take them on priority. That means current rule need to be optimized.
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
                f.write(f'Rule Advisor Message:\n{output_message}\n---------------------------------------------------------------\n')
                
            time.sleep(10)
            
            with open(f'{state.output_dir}/whole_log.txt','a') as f:
                f.write(f'Completion_tokens:\n{chat_completion.usage.completion_tokens}\nPrompt_tokens:\n{chat_completion.usage.prompt_tokens}---------------------------------------------------------------\n')    
            completion_tokens = state.completion_tokens + chat_completion.usage.completion_tokens
            prompt_tokens = state.prompt_tokens + chat_completion.usage.prompt_tokens
            return {'messages':[BaseMessage(content=output_message,sender='Rule Advisor')],'completion_tokens':completion_tokens,'prompt_tokens':prompt_tokens}
        
        except TypeError:
            time.sleep(10)
            print('GPT RETURN ERROR, Rule Advisor')