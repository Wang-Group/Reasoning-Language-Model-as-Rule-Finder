import os
import time

import json
import numpy as np
import pandas as pd
import pkg_resources
from pydantic import BaseModel, Field

from agent.state import AgentState,BaseMessage
from agent.client import client
from agent.json_paser import parse_LLM_json



class code(BaseModel):
    """Schema for code solutions for rule2matrix"""
    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")


def parse_llm_json(output_message):
    """
    Parse the LLM output JSON and generate the corresponding DataFrame
    """
    try:
        code_dict = json.loads(output_message)
        return code_dict
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}")
    
def remove_think_content(input_text):
    """
    Removes content within <think> tags, including the tags themselves, from the input string.

    Args:
        input_text (str): The text containing <think> tags.

    Returns:
        str: The input text with <think> tags and their content removed.
    """
    import re
    # Use regex to match and remove <think>...</think> and any enclosed content
    return re.sub(r'<think>.*?</think>', '', input_text, flags=re.DOTALL)

def Matrix_Generator(state:AgentState):
    '''Directly generate rule matrix based on the linguistic rules'''
    GPT_model = state.GPT_model
    GPT_seed = state.GPT_seed
    current_mtx_gen = state.current_mtx_gen
    # GPT_temperature = state.GPT_temperature
    train_path = os.path.join(state.output_dir,state.train_file)
    train_set = pd.read_csv(train_path)[['name','SMILES']]#read name and SMILES
    test_path = os.path.join(state.output_dir,state.test_file)
    test_set = pd.read_csv(test_path)[['name','SMILES']]#read name and SMILES
    # whole_SMILES = pd.concat([train_set,test_set],axis=0)
    whole_identifier = pd.concat([test_set,train_set],axis=0)#!Try test set first
    whole_SMILES = list(whole_identifier['SMILES'])
    train_len = len(train_set)
    whole_len = len(whole_SMILES)
    suggestions = '\n'
    if '** Start of Suggestions **' in state.messages[-1].content and state.messages[-1].sender == 'Matrix Checker':
        suggestions = suggestions + state.messages[-1].content
    
    with open(f'{state.output_dir}/current_rules.txt','r') as f:
        current_rules = f.read()
    reaction_background = state.reaction_background
    
    system_prompt = '''
    You are tasked with generating a feature matrix that contains only the values 0, 1, and -1 based on the given rules.
    If there are any suggestions, it means your transformation is being questioned by another agent. Please prioritize considering these suggestions.
    '''
    user_prompt = f'''
    !!Reaction Background!!
    {reaction_background}
    -------------------------------------------------------------------------------------------------
    
    !!Current Rules!!
    {current_rules}
    ------------------------------------------------------------------------------------------------
    
    !!SMILES Set!!
    {whole_SMILES}
    -------------------------------------------------------------------------------------------------
    
    !!Your Target!!
    Generate a feature matrix with the following criteria:
    - A value of 0 if the structural description of the rule does not match the modifier.
    - A value of 1 if the structural description of the rule matches the modifier and predicts a high target value.
    - A value of -1 if the structural description of the rule matches the modifier and predicts a low target value.
    Please generate the matrix rule by rule and summarize your response as a JSON object, with each rule as keys and the feature values as values. Ensure that the feature length for each rule is {whole_len}.
    ---------------------------------------------------------------------------------------------------------------
    
    !! Output Format !!
    Summarize your result into a JSON object.
    - The length of each rule's features should be {whole_len}.
    - The feature values should be only 0, 1, or -1.

    For example:
    {{
        "Rule 1:...": [list of corresponding features for the modifiers],
        "Rule 2:...": [list of corresponding features for the modifiers],
        ...
    }}
    
    Note: Strictly generate effective json format
    -------------------------------------------------------------------------------
    {suggestions}
    '''
    
    
    # print(system_prompt)
    # print(user_prompt)
    # while True:
    for i in range(100):
        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                model=GPT_model,
                temperature = 0.2,
                seed = GPT_seed,
                top_p = 0.1,
                n=1
            )# low creativity for matrix generation
            output_message = chat_completion.choices[0].message.content.strip()
            
            # Convert to DataFrame
            output_matrix = parse_LLM_json(output_message)
            df = pd.DataFrame(json.loads(output_matrix, strict=True))
            # train_matrix = df.iloc[:train_len,:]
            # test_matrix = df.iloc[train_len:,:]
            #!Try test set first
            train_matrix = df.iloc[(whole_len-train_len):,:]
            test_matrix = df.iloc[:(whole_len-train_len),:]
            train_matrix_file = 'train_matrix.csv'
            test_matrix_file = 'test_matrix.csv'
            train_matrix.to_csv(os.path.join(state.output_dir,train_matrix_file),index=None)
            test_matrix.to_csv(os.path.join(state.output_dir,test_matrix_file),index=None)
            # print(output_message)
            
            with open(f'{state.output_dir}/whole_log.txt','a') as f:
                f.write(f'Matrix Generator Message:\n{output_matrix}\n---------------------------------------------------------------\n')
            with open(f'{state.output_dir}/{state.current_matrix}','w') as f:
                f.write(f'Matrix Generator Message:\n{output_matrix}\n---------------------------------------------------------------\n')
                
            current_mtx_gen +=1
            time.sleep(10)
            
            return {'messages':[BaseMessage(content=output_matrix,sender='Matrix Generator')],'train_len':train_len,'whole_len':whole_len,'train_matrix':train_matrix_file,'test_matrix':test_matrix_file,'current_gen_count':0,'current_mtx_gen':current_mtx_gen}
        
        except TypeError or AttributeError:
            time.sleep(100)
            print('GPT RETURN ERROR, Matrix Generator')
        except Exception as e:
            print(e)
            return {'messages':[BaseMessage(content=f'**Matrix Generate Error**, please reconsider the rules.\nError:\n{e}',sender='Matrix Generator')],'current_gen_count':2,'current_mtx_gen':3}
        
def Matrix_Generator_o1(state:AgentState):
    '''Directly generate rule matrix based on the linguistic rules'''
    GPT_model = state.GPT_model
    GPT_seed = state.GPT_seed
    current_mtx_gen = state.current_mtx_gen
    # GPT_temperature = state.GPT_temperature
    train_path = os.path.join(state.output_dir,state.train_file)
    train_set = pd.read_csv(train_path)[['name','SMILES']]#read name and SMILES
    test_path = os.path.join(state.output_dir,state.test_file)
    test_set = pd.read_csv(test_path)[['name','SMILES']]#read name and SMILES
    # whole_SMILES = pd.concat([train_set,test_set],axis=0)
    whole_identifier = pd.concat([test_set,train_set],axis=0)#!Try test set first
    whole_SMILES = list(whole_identifier['SMILES'])
    train_len = len(train_set)
    whole_len = len(whole_SMILES)
    suggestions = '\n'
    if '** Start of Suggestions **' in state.messages[-1].content and state.messages[-1].sender == 'Matrix Checker':
        suggestions = suggestions + state.messages[-1].content
    
    with open(f'{state.output_dir}/current_rules.txt','r') as f:
        current_rules = f.read()
    reaction_background = state.reaction_background
    
    system_prompt = '''
    You are tasked with generating a feature matrix that contains only the values 0, 1, and -1 based on the given rules.
    If there are any suggestions, it means your transformation is being questioned by another agent. Please prioritize considering these suggestions.
    '''
    user_prompt = f'''
    !!Reaction Background!!
    {reaction_background}
    -------------------------------------------------------------------------------------------------
    
    !!Current Rules!!
    {current_rules}
    ------------------------------------------------------------------------------------------------
    
    !!SMILES Set!!
    {whole_SMILES}
    -------------------------------------------------------------------------------------------------
    
    !!Your Target!!
    Generate a feature matrix with the following criteria:
    - A value of 0 if the structural description of the rule does not match the modifier.
    - A value of 1 if the structural description of the rule matches the modifier and predicts a high target value.
    - A value of -1 if the structural description of the rule matches the modifier and predicts a low target value.
    Please generate the matrix rule by rule and summarize your response as a JSON object, with each rule as keys and the feature values as values. Ensure that the feature length for each rule is {whole_len}.
    ---------------------------------------------------------------------------------------------------------------
    
    !! Output Format !!
    Summarize your result into a JSON object.
    - The length of each rule's features should be {whole_len}.
    - The feature values should be only 0, 1, or -1.

    For example:
    {{
        "Rule 1:...": [list of corresponding features for the modifiers],
        "Rule 2:...": [list of corresponding features for the modifiers],
        ...
    }}
    
    Note: Strictly generate effective json format
    -------------------------------------------------------------------------------
    {suggestions}
    '''
    
    combined_prompt = system_prompt + '\n----------------------------------------\n' + user_prompt
    # print(system_prompt)
    # print(user_prompt)
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
            
            # Convert to DataFrame
            output_matrix = parse_LLM_json(output_message)
            df = pd.DataFrame(json.loads(output_matrix, strict=True))
            # train_matrix = df.iloc[:train_len,:]
            # test_matrix = df.iloc[train_len:,:]
            #!Try test set first
            train_matrix = df.iloc[(whole_len-train_len):,:]
            test_matrix = df.iloc[:(whole_len-train_len),:]
            train_matrix_file = 'train_matrix.csv'
            test_matrix_file = 'test_matrix.csv'
            train_matrix.to_csv(os.path.join(state.output_dir,train_matrix_file),index=None)
            test_matrix.to_csv(os.path.join(state.output_dir,test_matrix_file),index=None)
            # print(output_message)
            
            with open(f'{state.output_dir}/whole_log.txt','a') as f:
                f.write(f'Matrix Generator Message:\n{output_matrix}\n---------------------------------------------------------------\n')
            with open(f'{state.output_dir}/{state.current_matrix}','w') as f:
                f.write(f'Matrix Generator Message:\n{output_matrix}\n---------------------------------------------------------------\n')
                
            current_mtx_gen +=1
            time.sleep(10)
            
            return {'messages':[BaseMessage(content=output_matrix,sender='Matrix Generator')],'train_len':train_len,'whole_len':whole_len,'train_matrix':train_matrix_file,'test_matrix':test_matrix_file,'current_gen_count':0,'current_mtx_gen':current_mtx_gen}
        
        except TypeError or AttributeError:
            time.sleep(100)
            print('GPT RETURN ERROR, Matrix Generator')
        except Exception as e:
            print(e)
            return {'messages':[BaseMessage(content=f'**Matrix Generate Error**, please reconsider the rules.\nError:\n{e}',sender='Matrix Generator')],'current_gen_count':2,'current_mtx_gen':3}
            
            
            
def Matrix_Checker(state:AgentState):
    '''Check if regeneration of rule matrix is needed.'''
    GPT_model = state.GPT_model
    GPT_seed = state.GPT_seed
    # GPT_temperature = state.GPT_temperature    
    train_path = os.path.join(state.output_dir,state.train_file)
    train_set = pd.read_csv(train_path)[['name','SMILES']]#read name and SMILES
    test_path = os.path.join(state.output_dir,state.test_file)
    test_set = pd.read_csv(test_path)[['name','SMILES']]#read name and SMILES
    # whole_SMILES = pd.concat([train_set,test_set],axis=0)
    whole_identifier = pd.concat([test_set,train_set],axis=0)#!Try test set first
    whole_SMILES = list(whole_identifier['SMILES'])
    data_matrix = state.messages[-1].content
    
    
    with open(f'{state.output_dir}/current_rules.txt','r') as f:
        current_rules = f.read()
    
    system_prompt = '''
    Your task is to check whether the transformation from language rules to a numeric feature matrix is effective.

    The feature matrix is generated with the following criteria:
    - A value of 0 if the structural description of the rule does not match the modifier.
    - A value of 1 if the structural description of the rule matches the modifier and predicts a high target value.
    - A value of -1 if the structural description of the rule matches the modifier and predicts a low target value.
    '''
    user_prompt = f'''
    !!Current Rules!!
    {current_rules}
    ------------------------------------------------------------------------------------------------
    
    !!SMILES Set!!
    {whole_SMILES}
    -------------------------------------------------------------------------------------------------
    
    !!Feature Matrix!!
    {data_matrix}
    -------------------------------------------------------------------------------------------------
    
    !!Your Target!!
    Check if the feature matrix fits the rules. Please check this carefully for each rule and each modifier. 
    Criteria:
    - Check the matrix rule by rule, SMILES by SMILES.
    - If there are many 0s in the feature matrix, verify if the rule truly does not match, especially for those 0s at the end of each rule feature list.
    - Ensure that the 1s/-1s accurately correspond to high/low target values according to each rule.
    - Take your time for thorough evaluation, but avoid making unfounded assumptions.
    
    If you find that the transformation is not successful, provide practical suggestions for your collaborator to regenerate the feature matrix.
    ------------------------------------------------------------------------------------------------------------------------------------------
    
    !!Format of suggestions!!
    ** Start of Suggestions **
    - ** Suggestion 1 **: ...
    - ** Suggestion 2 **: ...
    - ...
    ** End of Suggestions **
    ----------------------------------------------------------------------------------------------------------------------------------------
    
    If the transformation from language rules to numeric feature matrix is effective, add '**TRUE**' at the end of your answer. Otherwise, do not add anything at the end.
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
                f.write(f'Matrix Checker Message:\n{output_message}\n---------------------------------------------------------------\n')
                
            time.sleep(10)

            return {'messages':[BaseMessage(content=output_message,sender='Matrix Checker')]}
        
        except TypeError or AttributeError:
            time.sleep(100)
            print('GPT RETURN ERROR, Matrix Checker')

def Matrix_Checker_o1(state:AgentState):
    '''Check if regeneration of rule matrix is needed.'''
    GPT_model = state.GPT_model
    GPT_seed = state.GPT_seed
    # GPT_temperature = state.GPT_temperature    
    train_path = os.path.join(state.output_dir,state.train_file)
    train_set = pd.read_csv(train_path)[['name','SMILES']]#read name and SMILES
    test_path = os.path.join(state.output_dir,state.test_file)
    test_set = pd.read_csv(test_path)[['name','SMILES']]#read name and SMILES
    # whole_SMILES = pd.concat([train_set,test_set],axis=0)
    whole_identifier = pd.concat([test_set,train_set],axis=0)#!Try test set first
    whole_SMILES = list(whole_identifier['SMILES'])
    data_matrix = state.messages[-1].content
    
    
    with open(f'{state.output_dir}/current_rules.txt','r') as f:
        current_rules = f.read()
    
    system_prompt = '''
    Your task is to check whether the transformation from language rules to a numeric feature matrix is effective.

    The feature matrix is generated with the following criteria:
    - A value of 0 if the structural description of the rule does not match the modifier.
    - A value of 1 if the structural description of the rule matches the modifier and predicts a high target value.
    - A value of -1 if the structural description of the rule matches the modifier and predicts a low target value.
    '''
    user_prompt = f'''
    !!Current Rules!!
    {current_rules}
    ------------------------------------------------------------------------------------------------
    
    !!SMILES Set!!
    {whole_SMILES}
    -------------------------------------------------------------------------------------------------
    
    !!Feature Matrix!!
    {data_matrix}
    -------------------------------------------------------------------------------------------------
    
    !!Your Target!!
    Check if the feature matrix fits the rules. Please check this carefully for each rule and each modifier. 
    Criteria:
    - Check the matrix rule by rule, SMILES by SMILES.
    - If there are many 0s in the feature matrix, verify if the rule truly does not match, especially for those 0s at the end of each rule feature list.
    - Ensure that the 1s/-1s accurately correspond to high/low target values according to each rule.
    - Take your time for thorough evaluation, but avoid making unfounded assumptions.
    
    If you find that the transformation is not successful, provide practical suggestions for your collaborator to regenerate the feature matrix.
    ------------------------------------------------------------------------------------------------------------------------------------------
    
    !!Format of suggestions!!
    ** Start of Suggestions **
    - ** Suggestion 1 **: ...
    - ** Suggestion 2 **: ...
    - ...
    ** End of Suggestions **
    ----------------------------------------------------------------------------------------------------------------------------------------
    
    If the transformation from language rules to numeric feature matrix is effective, add '**TRUE**' at the end of your answer. Otherwise, do not add anything at the end.
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
                top_p=1,
                n=1
            )
            output_message = chat_completion.choices[0].message.content.strip()
            with open(f'{state.output_dir}/whole_log.txt','a') as f:
                f.write(f'Matrix Checker Message:\n{output_message}\n---------------------------------------------------------------\n')
                
            time.sleep(10)

            return {'messages':[BaseMessage(content=output_message,sender='Matrix Checker')]}
        
        except TypeError or AttributeError:
            time.sleep(100)
            print('GPT RETURN ERROR, Matrix Checker')
    
def Coded_Matrix_Generator(state:AgentState):
    GPT_model = state.GPT_model
    GPT_seed = state.GPT_seed
    current_mtx_gen = state.current_mtx_gen
    # GPT_temperature = state.GPT_temperature
    train_path = os.path.join(state.output_dir,state.train_file)
    train_set = pd.read_csv(train_path)[['name','SMILES']]#read name and SMILES
    test_path = os.path.join(state.output_dir,state.test_file)
    test_set = pd.read_csv(test_path)[['name','SMILES']]#read name and SMILES
    # whole_SMILES = pd.concat([train_set,test_set],axis=0)
    whole_identifier = pd.concat([test_set,train_set],axis=0)#!Try test set first
    whole_SMILES = list(whole_identifier['SMILES'])
    train_len = len(train_set)
    whole_len = len(whole_SMILES)
    
    suggestions = '\n'
    if '** Start of Suggestions **' in state.messages[-1].content and state.messages[-1].sender == 'Matrix Checker':
        suggestions = suggestions + state.messages[-1].content
        with open(f'{state.output_dir}/current_rule_code.txt','r') as f:
            current_rule_code = f.read()
        suggestions  = suggestions +f'''--------------------------------------------------------------------------\n
        !!Current code!!
        {current_rule_code}'''
        
    system_prompt = '''You are a coding assistant with expertise in RDkit. Your task is to generate Python code that takes a list of SMILES strings as input. The code should follow the provided natural language rules to convert these SMILES strings into a feature matrix using RDkit. The output matrix should be a DataFrame where each column corresponds to one rule, and each row corresponds to one SMILES string from the list. There should be number_of_SMILES rows and number_of_rules columns.
    Generate a feature matrix with the following criteria:
    - A value of 0 if the structural description of the rule does not match the SMILES.
    - A value of 1 if the structural description of the rule matches the SMILES and predicts a high target value.
    - A value of -1 if the structural description of the rule matches the SMILES and predicts a low target value.
    '''
    with open('/home/lnh/loffi.txt','r') as f:
        smarts_intro = f.read()

    with open(f'{state.output_dir}/current_rules.txt','r') as f:
        current_rules = f.read()
        
    user_prompt = f'''
    !!Examples for SMARTS!!
    {smarts_intro}
    -----------------------------------------------------------------------------------------------

    !!Current Rules!!
    {current_rules}
    ------------------------------------------------------------------------------------------------
    
    !!Suggestions from Matrix Checker!!
    {suggestions}
    -------------------------------------------------------------------------------------------------

    Please generate Python code that follows these rules. 
    Your code should be structured in the following format:

    {{
        "prefix": "<Description of the problem and approach>",
        "imports": "<Code block containing import statements>",
        "code": "<Code block not including import statements>"
    }}

    Example:

    {{
        "prefix": "This code converts a list of SMILES strings into a feature matrix using RDkit.",
        "imports": "import pandas as pd\\nfrom rdkit import Chem\\nfrom rdkit.Chem import AllChem",
        "code": "def rule2matrix(smiles_list):\\n    rules = [ ['[CX3](=O)[OX2H1]', '[#7]',], # Rule 1 ['[CX3](=O)[OX2H1]', '[C;X4][C;X4][C;X4][C;X4][C;X4][C;X4]',], # Rule 2 ['[c][CX3](=O)[OX2H1]',], # Rule 3 ...]\\n    results = []\\n  
        for smi in smiles_list:  \\n
        mol = Chem.MolFromSmiles(smi) if mol is None: \\n results.append([0] * len(rules)) \\n continue
        row = [] for i, rule in enumerate(rules): try: if all(mol.HasSubstructMatch(Chem.MolFromSmarts(r)) for r in rule): if i in [0, 2, 3, 4, 5, ...]:#Rules with high prediction \\n row.append(1) \\n else: \\n row.append(-1) \\n else: \\n row.append(0) \\n except: \n row.append(0)
        results.append(row)
        df = pd.DataFrame(results, columns=[f'Rule {{i+1}}' for i in range(len(rules))])
        return df"
    }}

    Note:
    Name the function as rule2matrix, Define the function without any example to run that function.
    Using SMARTS for better substructure search.
    Handle possible error: when there is any error for one rule apply to one SMILES, return 0 instead.
    '''
    
    
    # print(system_prompt)
    # print(user_prompt)
    # while True:
    for i in range(10):
        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                model=GPT_model,
                temperature = 0.2,
                seed = GPT_seed,
                top_p = 0.1,
                n=1
            )
            output_message = chat_completion.choices[0].message.content.strip()
            code_dict = parse_llm_json(parse_LLM_json(output_message))
            code_solution = code(**code_dict)
            exec(code_solution.imports, globals())
            exec(code_solution.code, globals())
            df = rule2matrix(whole_SMILES) # type: ignore
            # rule_code = json.dumps(code_dict)
            # rule_code = output_message
            rule_code = str(code_solution.code)
            output_matrix = df.to_csv(index=False)
            # train_matrix = df.iloc[:train_len,:]
            # test_matrix = df.iloc[train_len:,:]
            #!Try test set first
            train_matrix = df.iloc[(whole_len-train_len):,:]
            test_matrix = df.iloc[:(whole_len-train_len),:]
            train_matrix_file = 'train_matrix.csv'
            test_matrix_file = 'test_matrix.csv'
            train_matrix.to_csv(os.path.join(state.output_dir,train_matrix_file),index=None)
            test_matrix.to_csv(os.path.join(state.output_dir,test_matrix_file),index=None)
            # print(output_message)
            with open(f'{state.output_dir}/current_rule_code.json', 'w') as f:
                json.dump(code_dict, f, indent=4)  
            with open(f'{state.output_dir}/current_rule_code.txt','w') as f:
                f.write(rule_code)
            with open(f'{state.output_dir}/whole_log.txt','a') as f:
                f.write(f'Code for rules:\n{rule_code}\n---------------------------------------------------------------\n')
            with open(f'{state.output_dir}/whole_log.txt','a') as f:
                f.write(f'Current rules:\n{current_rules}\n---------------------------------------------------------------\n')
            with open(f'{state.output_dir}/whole_log.txt','a') as f:
                f.write(f'Matrix Generator Message:\n{output_matrix}\n---------------------------------------------------------------\n')
            with open(f'{state.output_dir}/{state.current_matrix}','w') as f:
                f.write(f'Matrix Generator Message:\n{output_matrix}\n---------------------------------------------------------------\n')
            current_mtx_gen +=1
            time.sleep(10)
            
            return {'messages':[BaseMessage(content=output_matrix,sender='Matrix Generator')],'train_len':train_len,'whole_len':whole_len,'train_matrix':train_matrix_file,'test_matrix':test_matrix_file,'current_gen_count':0,'current_mtx_gen':current_mtx_gen}
        
        except TypeError or AttributeError:
            time.sleep(100)
            print('GPT RETURN ERROR, Matrix Generator')
        except Exception as e:
            print(e)
            return {'messages':[BaseMessage(content=f'**Matrix Generate Error**, please reconsider the rules.\nError:\n{e}',sender='Matrix Generator')],'current_gen_count':2,'current_mtx_gen':3}
        
# def Coded_Matrix_Generator(state:AgentState):
#     '''Convert linguistic rules to rule matrix via GPT-generated, SMARTS based code.'''
#     GPT_model = state.GPT_model
#     GPT_seed = state.GPT_seed
#     current_mtx_gen = state.current_mtx_gen
#     # GPT_temperature = state.GPT_temperature
#     train_path = os.path.join(state.output_dir,state.train_file)
#     train_set = pd.read_csv(train_path)[['name','SMILES']]#read name and SMILES
#     test_path = os.path.join(state.output_dir,state.test_file)
#     test_set = pd.read_csv(test_path)[['name','SMILES']]#read name and SMILES
#     # whole_SMILES = pd.concat([train_set,test_set],axis=0)
#     whole_identifier = pd.concat([test_set,train_set],axis=0)#!Try test set first
#     whole_SMILES = list(whole_identifier['SMILES'])
#     train_len = len(train_set)
#     whole_len = len(whole_SMILES)
    
#     with open(pkg_resources.resource_filename('agent.data', 'rule_code_eg.txt'),'r') as f:
#         code_example = f.read()

#     suggestions = '\n'
#     if '** Start of Suggestions **' in state.messages[-1].content and state.messages[-1].sender == 'Matrix Checker':
#         suggestions = suggestions + state.messages[-1].content
#         with open(f'{state.output_dir}/current_rule_code.txt','r') as f:
#             current_rule_code = f.read()
#         suggestions  = suggestions +f'''--------------------------------------------------------------------------\n
#         !!Current code!!
#         {current_rule_code}'''
        
#     system_prompt = '''You are a coding assistant with expertise in RDkit. Your task is to generate Python code that takes a list of SMILES strings as input. The code should follow the provided natural language rules to convert these SMILES strings into a feature matrix using RDkit. The output matrix should be a DataFrame where each column corresponds to one rule, and each row corresponds to one SMILES string from the list. There should be number_of_SMILES rows and number_of_rules columns.
#     Generate a feature matrix with the following criteria:
#     - A value of 0 if the structural description of the rule does not match the SMILES.
#     - A value of 1 if the structural description of the rule matches the SMILES and predicts a high target value.
#     - A value of -1 if the structural description of the rule matches the SMILES and predicts a low target value.
#     '''
#     with open(pkg_resources.resource_filename('agent.data', 'loffi.txt'),'r') as f:
#         smarts_intro = f.read()

#     with open(f'{state.output_dir}/current_rules.txt','r') as f:
#         current_rules = f.read()
    
#     with open(pkg_resources.resource_filename('agent.data', 'group_examples.txt'),'r') as f:
#         group_egs = f.read()
#     user_prompt = f'''
#     !!Examples for SMARTS!!
#     {smarts_intro}
#     {group_egs}
#     -----------------------------------------------------------------------------------------------

#     !!Current Rules!!
#     {current_rules}
#     ------------------------------------------------------------------------------------------------
    
#     !!Suggestions from Matrix Checker!!
#     {suggestions}
#     -------------------------------------------------------------------------------------------------

#     Please generate Python code that follows these rules. 
#     Your code should be structured in the following format:

#     Example:

#     {{
#         "prefix": "This code converts a list of SMILES strings into a feature matrix using RDkit.",
#         "imports": "import pandas as pd\\nfrom rdkit import Chem\\nfrom rdkit.Chem import AllChem",
#         "code": "def rule2matrix(smiles_list):\\n    rules = [ ...  \\n {{'name': 'Rule 10','patterns': ['[#7,#8,#16]',  # Multiple donor atoms (N, O, S)'[CX3](=O)[OX2H1]',  # Carboxylic acid group],'count': {{'[#7,#8,#16]': 2}},  # At least two donor atoms \\n 'exclude': ...,   \\n'prediction': 1,...
#     }},
# ]\\n    results = []\\n  
#         for smi in smiles_list:  \\n
#         \\n
#         results.append(row)
#         df = pd.DataFrame(results, columns=[f'Rule {{i+1}}' for i in range(len(rules))])
#         return df"
#     }}

#     Note:
#     Name the function as rule2matrix, Define the function without any example to run that function.
#     Using SMARTS for better substructure search.
#     Handle possible error: when there is any error for one rule apply to one SMILES, return 0 instead.
#     '''
    
    
#     # print(system_prompt)
#     # print(user_prompt)
#     # while True:
#     for i in range(10):
#         try:
#             chat_completion = client.chat.completions.create(
#                 messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
#                 model=GPT_model,
#                 temperature = 0.2,
#                 seed = GPT_seed,
#                 top_p = 0.1,
#                 n=1
#             )
#             output_message = chat_completion.choices[0].message.content.strip()
#             code_dict = parse_llm_json(parse_LLM_json(output_message))
#             code_solution = code(**code_dict)
#             exec(code_solution.imports, globals())
#             exec(code_solution.code, globals())
#             df = rule2matrix(whole_SMILES) # type: ignore
#             # rule_code = json.dumps(code_dict)
#             # rule_code = output_message
#             rule_code = str(code_solution.code)
#             output_matrix = df.to_csv(index=False)
#             # train_matrix = df.iloc[:train_len,:]
#             # test_matrix = df.iloc[train_len:,:]
#             #!Try test set first
#             train_matrix = df.iloc[(whole_len-train_len):,:]
#             test_matrix = df.iloc[:(whole_len-train_len),:]
#             train_matrix_file = 'train_matrix.csv'
#             test_matrix_file = 'test_matrix.csv'
#             train_matrix.to_csv(os.path.join(state.output_dir,train_matrix_file),index=None)
#             test_matrix.to_csv(os.path.join(state.output_dir,test_matrix_file),index=None)
#             # print(output_message)
#             with open(f'{state.output_dir}/current_rule_code.json', 'w') as f:
#                 json.dump(code_dict, f, indent=4)  
#             with open(f'{state.output_dir}/current_rule_code.txt','w') as f:
#                 f.write(rule_code)
#             with open(f'{state.output_dir}/whole_log.txt','a') as f:
#                 f.write(f'Code for rules:\n{rule_code}\n---------------------------------------------------------------\n')
#             with open(f'{state.output_dir}/whole_log.txt','a') as f:
#                 f.write(f'Current rules:\n{current_rules}\n---------------------------------------------------------------\n')
#             with open(f'{state.output_dir}/whole_log.txt','a') as f:
#                 f.write(f'Matrix Generator Message:\n{output_matrix}\n---------------------------------------------------------------\n')
#             with open(f'{state.output_dir}/{state.current_matrix}','w') as f:
#                 f.write(f'Matrix Generator Message:\n{output_matrix}\n---------------------------------------------------------------\n')
#             current_mtx_gen +=1
#             time.sleep(10)
            
#             return {'messages':[BaseMessage(content=output_matrix,sender='Matrix Generator')],'train_len':train_len,'whole_len':whole_len,'train_matrix':train_matrix_file,'test_matrix':test_matrix_file,'current_gen_count':0,'current_mtx_gen':current_mtx_gen}
        
#         except TypeError:
#             time.sleep(100)
#             print('GPT RETURN ERROR, Matrix Generator')
#         except Exception as e:
#             print(e)
#             return {'messages':[BaseMessage(content=f'**Matrix Generate Error**, please reconsider the rules.\nError:\n{e}',sender='Matrix Generator')],'current_gen_count':2,'current_mtx_gen':3}



def Coded_Matrix_Checker(state:AgentState):
    '''Check if regeneration of rule code, therefore the rule matrix, is needed'''
    GPT_model = state.GPT_model
    GPT_seed = state.GPT_seed
    # GPT_temperature = state.GPT_temperature    
    train_path = os.path.join(state.output_dir,state.train_file)
    train_set = pd.read_csv(train_path)[['name','SMILES']]#read name and SMILES
    test_path = os.path.join(state.output_dir,state.test_file)
    test_set = pd.read_csv(test_path)[['name','SMILES']]#read name and SMILES
    # whole_SMILES = pd.concat([train_set,test_set],axis=0)
    whole_identifier = pd.concat([test_set,train_set],axis=0)#!Try test set first
    whole_SMILES = list(whole_identifier['SMILES'])
    data_matrix = state.messages[-1].content
    
    
    with open(f'{state.output_dir}/current_rules.txt','r') as f:
        current_rules = f.read()
        
        
    with open(pkg_resources.resource_filename('agent.data', 'loffi.txt'),'r') as f:
        smarts_intro = f.read()
    with open(f'{state.output_dir}/current_rule_code.txt','r') as f:
        rule_code = f.read()
    system_prompt = '''
    Your task is to check whether the transformation from language rules to a numeric feature matrix is effective. The matrix is generated by a code written by Matrix Generator. If the matrix transformation is not effective, please give suggestions to improve current code.

    The feature matrix is generated with the following criteria:
    - A value of 0 if the structural description of the rule does not match the modifier.
    - A value of 1 if the structural description of the rule matches the modifier and predicts a high target value.
    - A value of -1 if the structural description of the rule matches the modifier and predicts a low target value.
    '''
    user_prompt = f'''
    !!Current Rules!!
    {current_rules}
    ------------------------------------------------------------------------------------------------
    
    !!SMILES Set!!
    {whole_SMILES}
    -------------------------------------------------------------------------------------------------
    
    !!Feature Matrix!!
    {data_matrix}
    -------------------------------------------------------------------------------------------------
    
    !!Examples for SMARTS!!
    {smarts_intro}
    -----------------------------------------------------------------------------------------------
    
    !!Current code!!
    {rule_code}
    ------------------------------------------------------------------------------------------------
    
    !!Your Target!!
    Check if the feature matrix fits the rules. Please check this carefully for each rule and each modifier. 
    Criteria:
    - Check the matrix rule by rule, SMILES by SMILES.
    - If there are many 0s in the feature matrix, verify if the rule truly does not match, especially for those 0s at the end of each rule feature list.
    - Ensure that the 1s/-1s accurately correspond to high/low target values according to each rule.
    - Take your time for thorough evaluation, but avoid making unfounded assumptions.
    
    Give suggestions for how to improve rule code write by Matrix Generator:
    - Carefully check the SMARTS write in the code based on if it describes the natural language rules correctly.
    - Give suggestions to improve the generated code
    If you find that the transformation is not successful, provide practical suggestions for your collaborator to regenerate the code generating rule matrix.
    ------------------------------------------------------------------------------------------------------------------------------------------
    
    !!Format of suggestions!!
    ** Start of Suggestions **
    - ** Suggestion 1 **: ...
    - ** Suggestion 2 **: ...
    - ...
    ** End of Suggestions **
    ----------------------------------------------------------------------------------------------------------------------------------------
    
    If the transformation from language rules to numeric feature matrix is effective, add '**TRUE**' at the end of your answer. Otherwise, do not add anything at the end.
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
                f.write(f'Matrix Checker Message:\n{output_message}\n---------------------------------------------------------------\n')
                
            time.sleep(10)

            return {'messages':[BaseMessage(content=output_message,sender='Matrix Checker')]}
        
        except TypeError or AttributeError:
            time.sleep(100)
            print('GPT RETURN ERROR, Matrix Checker')
      

            
            
            
            
def Coded_Matrix_Generator_o1(state:AgentState):
    '''Adjust GPT-invocation for gpt-o1'''

    GPT_model = state.GPT_model
    GPT_seed = state.GPT_seed
    current_mtx_gen = state.current_mtx_gen
    # GPT_temperature = state.GPT_temperature
    train_path = os.path.join(state.output_dir,state.train_file)
    train_set = pd.read_csv(train_path)[['name','SMILES']]#read name and SMILES
    test_path = os.path.join(state.output_dir,state.test_file)
    test_set = pd.read_csv(test_path)[['name','SMILES']]#read name and SMILES
    # whole_SMILES = pd.concat([train_set,test_set],axis=0)
    whole_identifier = pd.concat([test_set,train_set],axis=0)#!Try test set first
    whole_SMILES = list(whole_identifier['SMILES'])
    train_len = len(train_set)
    whole_len = len(whole_SMILES)
    
    with open(pkg_resources.resource_filename('agent.data', 'rule_code_eg.txt'),'r') as f:
        code_example = f.read()
    suggestions = '\n'
    if '** Start of Suggestions **' in state.messages[-1].content and state.messages[-1].sender == 'Matrix Checker':
        suggestions = suggestions + state.messages[-1].content
        with open(f'{state.output_dir}/current_rule_code.txt','r') as f:
            current_rule_code = f.read()
        suggestions  = suggestions +f'''--------------------------------------------------------------------------\n
        !!Current code!!
        {current_rule_code}'''
        
    system_prompt = '''You are a coding assistant with expertise in RDkit. Your task is to generate Python code that takes a list of SMILES strings as input. The code should follow the provided natural language rules to convert these SMILES strings into a feature matrix using RDkit. The output matrix should be a DataFrame where each column corresponds to one rule, and each row corresponds to one SMILES string from the list. There should be number_of_SMILES rows and number_of_rules columns.
    Generate a feature matrix with the following criteria:
    - A value of 0 if the structural description of the rule does not match the SMILES.
    - A value of 1 if the structural description of the rule matches the SMILES and predicts a high target value.
    - A value of -1 if the structural description of the rule matches the SMILES and predicts a low target value.
    '''
    with open(pkg_resources.resource_filename('agent.data', 'loffi.txt'),'r') as f:
        smarts_intro = f.read()
    with open(pkg_resources.resource_filename('agent.data', 'MACCS_examples.txt'),'r') as f:
        MACCS_egs = f.read()
    with open(f'{state.output_dir}/current_rules.txt','r') as f:
        current_rules = f.read()
    with open(pkg_resources.resource_filename('agent.data', 'group_examples.txt'),'r') as f:
        group_egs = f.read()
            
    user_prompt = f'''
    !!Examples for SMARTS!!
    {smarts_intro}
    {group_egs}
    -----------------------------------------------------------------------------------------------

    !!Current Rules!!
    {current_rules}
    ------------------------------------------------------------------------------------------------
    
    !!Suggestions from Matrix Checker!!
    {suggestions}
    -------------------------------------------------------------------------------------------------

    Please generate Python code that follows these rules. 
    Your code should be structured in the following format:

    {{
        "prefix": "<Description of the problem and approach>",
        "imports": "<Code block containing import statements>",
        "code": "<Code block not including import statements>"
    }}

    Example for "code":
    {{
    "prefix": "This code converts a list of SMILES strings into a feature matrix using RDkit.",
    "imports": "import pandas as pd\\nfrom rdkit import Chem\\nfrom rdkit.Chem import AllChem",
    "code": {code_example}
    \\n 
    }}
    
    

    Note:
    Name the function as rule2matrix, Define the function without any example to run that function.
    Using SMARTS for better substructure search.
    Consider appropriate logic (and, or, not/exclude) of SMARTS patterns to describe a rule.
    Handle possible error: when there is any error for one rule apply to one SMILES, return 0 instead.
    '''
    #! Should the example containing all(mol.HasSubstructMatch(Chem.MolFromSmarts(r)) for r in rule)?
    combined_prompt = system_prompt + '\n----------------------------------------\n' + user_prompt
    
    # print(system_prompt)
    # print(user_prompt)
    # while True:
    for i in range(10):
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
            code_dict = parse_llm_json(parse_LLM_json(output_message))
            code_solution = code(**code_dict)
            rule_code = str(code_solution.code)
            with open(f'{state.output_dir}/current_rule_code.txt','w') as f:
                f.write(rule_code)
            exec(code_solution.imports, globals())
            exec(code_solution.code, globals())
            df = rule2matrix(whole_SMILES) # type: ignore
            # rule_code = json.dumps(code_dict)
            # rule_code = output_message
            output_matrix = df.to_csv(index=False)
            # train_matrix = df.iloc[:train_len,:]
            # test_matrix = df.iloc[train_len:,:]
            #!Try test set first
            train_matrix = df.iloc[(whole_len-train_len):,:]
            test_matrix = df.iloc[:(whole_len-train_len),:]
            train_matrix_file = 'train_matrix.csv'
            test_matrix_file = 'test_matrix.csv'
            train_matrix.to_csv(os.path.join(state.output_dir,train_matrix_file),index=None)
            test_matrix.to_csv(os.path.join(state.output_dir,test_matrix_file),index=None)
            # print(output_message)
            with open(f'{state.output_dir}/current_rule_code.json', 'w') as f:
                json.dump(code_dict, f, indent=4) 
            with open(f'{state.output_dir}/whole_log.txt','a') as f:
                f.write(f'Code for rules:\n{rule_code}\n---------------------------------------------------------------\n')
            with open(f'{state.output_dir}/whole_log.txt','a') as f:
                f.write(f'Current rules:\n{current_rules}\n---------------------------------------------------------------\n')
            with open(f'{state.output_dir}/whole_log.txt','a') as f:
                f.write(f'Matrix Generator Message:\n{output_matrix}\n---------------------------------------------------------------\n')
            with open(f'{state.output_dir}/{state.current_matrix}','w') as f:
                f.write(f'Matrix Generator Message:\n{output_matrix}\n---------------------------------------------------------------\n')
            current_mtx_gen +=1
            time.sleep(10)
            
            with open(f'{state.output_dir}/whole_log.txt','a') as f:
                f.write(f'Completion_tokens:\n{chat_completion.usage.completion_tokens}\nPrompt_tokens:\n{chat_completion.usage.prompt_tokens}---------------------------------------------------------------\n')    
            completion_tokens = state.completion_tokens + chat_completion.usage.completion_tokens
            prompt_tokens = state.prompt_tokens + chat_completion.usage.prompt_tokens
            
            return {'messages':[BaseMessage(content=output_matrix,sender='Matrix Generator')],'train_len':train_len,'whole_len':whole_len,'train_matrix':train_matrix_file,'test_matrix':test_matrix_file,'current_gen_count':0,'current_mtx_gen':current_mtx_gen,'completion_tokens':completion_tokens,'prompt_tokens':prompt_tokens}
        
        except TypeError or AttributeError:
            time.sleep(10)
            print('GPT RETURN ERROR, Matrix Generator')
        except Exception as e:
            print(e)
            return {'messages':[BaseMessage(content=f'**Matrix Generate Error**, please reconsider the rules.\nError:\n{e}',sender='Matrix Generator')],'current_gen_count':2,'current_mtx_gen':3}



def Coded_Matrix_Generator_DS(state:AgentState):
    '''Adjust GPT-invocation for deepseek'''

    GPT_model = state.GPT_model
    GPT_seed = state.GPT_seed
    current_mtx_gen = state.current_mtx_gen
    # GPT_temperature = state.GPT_temperature
    train_path = os.path.join(state.output_dir,state.train_file)
    train_set = pd.read_csv(train_path)[['name','SMILES']]#read name and SMILES
    test_path = os.path.join(state.output_dir,state.test_file)
    test_set = pd.read_csv(test_path)[['name','SMILES']]#read name and SMILES
    # whole_SMILES = pd.concat([train_set,test_set],axis=0)
    whole_identifier = pd.concat([test_set,train_set],axis=0)#!Try test set first
    whole_SMILES = list(whole_identifier['SMILES'])
    train_len = len(train_set)
    whole_len = len(whole_SMILES)
    
    with open(pkg_resources.resource_filename('agent.data', 'rule_code_eg.txt'),'r') as f:
        code_example = f.read()
    suggestions = '\n'
    if '** Start of Suggestions **' in state.messages[-1].content and state.messages[-1].sender == 'Matrix Checker':
        suggestions = suggestions + state.messages[-1].content
        with open(f'{state.output_dir}/current_rule_code.txt','r') as f:
            current_rule_code = f.read()
        suggestions  = suggestions +f'''--------------------------------------------------------------------------\n
        !!Current code!!
        {current_rule_code}'''
        
    system_prompt = '''You are a coding assistant with expertise in RDkit. Your task is to generate Python code that takes a list of SMILES strings as input. The code should follow the provided natural language rules to convert these SMILES strings into a feature matrix using RDkit. The output matrix should be a DataFrame where each column corresponds to one rule, and each row corresponds to one SMILES string from the list. There should be number_of_SMILES rows and number_of_rules columns.
    Generate a feature matrix with the following criteria:
    - A value of 0 if the structural description of the rule does not match the SMILES.
    - A value of 1 if the structural description of the rule matches the SMILES and predicts a high target value.
    - A value of -1 if the structural description of the rule matches the SMILES and predicts a low target value.
    '''
    with open(pkg_resources.resource_filename('agent.data', 'loffi.txt'),'r') as f:
        smarts_intro = f.read()
    with open(pkg_resources.resource_filename('agent.data', 'MACCS_examples.txt'),'r') as f:
        MACCS_egs = f.read()
    with open(f'{state.output_dir}/current_rules.txt','r') as f:
        current_rules = f.read()
    with open(pkg_resources.resource_filename('agent.data', 'group_examples.txt'),'r') as f:
        group_egs = f.read()
            
    user_prompt = f'''
    !!Examples for SMARTS!!
    {smarts_intro}
    {group_egs}
    -----------------------------------------------------------------------------------------------

    !!Current Rules!!
    {current_rules}
    ------------------------------------------------------------------------------------------------
    
    !!Suggestions from Matrix Checker!!
    {suggestions}
    -------------------------------------------------------------------------------------------------

    Please generate Python code that follows these rules. 
    Your code should be structured in the following format:

    {{
        "prefix": "<Description of the problem and approach>",
        "imports": "<Code block containing import statements>",
        "code": "<Code block not including import statements>"
    }}

    Example for "code":
    {{
    "prefix": "This code converts a list of SMILES strings into a feature matrix using RDkit.",
    "imports": "import pandas as pd\\nfrom rdkit import Chem\\nfrom rdkit.Chem import AllChem",
    "code": {code_example}
    \\n 
    }}
    
    

    Note:
    Name the function as rule2matrix, Define the function without any example to run that function.
    Using SMARTS for better substructure search.
    Consider appropriate logic (and, or, not/exclude) of SMARTS patterns to describe a rule.
    Handle possible error: when there is any error for one rule apply to one SMILES, return 0 instead.
    '''
    #! Should the example containing all(mol.HasSubstructMatch(Chem.MolFromSmarts(r)) for r in rule)?
    combined_prompt = system_prompt + '\n----------------------------------------\n' + user_prompt
    
    # print(system_prompt)
    # print(user_prompt)
    # while True:
    for i in range(10):
        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": combined_prompt}],
                model=GPT_model,
                temperature = 0,
                seed = GPT_seed,
                top_p = 1,
                n=1
            )
            output_message = chat_completion.choices[0].message.content.strip()
            clean_text = remove_think_content(output_message).strip()
            code_dict = parse_llm_json(parse_LLM_json(clean_text))
            code_solution = code(**code_dict)
            rule_code = str(code_solution.code)
            with open(f'{state.output_dir}/current_rule_code.txt','w') as f:
                f.write(rule_code)
            exec(code_solution.imports, globals())
            exec(code_solution.code, globals())
            df = rule2matrix(whole_SMILES) # type: ignore
            # rule_code = json.dumps(code_dict)
            # rule_code = output_message
            
            output_matrix = df.to_csv(index=False)
            # train_matrix = df.iloc[:train_len,:]
            # test_matrix = df.iloc[train_len:,:]
            #!Try test set first
            train_matrix = df.iloc[(whole_len-train_len):,:]
            test_matrix = df.iloc[:(whole_len-train_len),:]
            train_matrix_file = 'train_matrix.csv'
            test_matrix_file = 'test_matrix.csv'
            train_matrix.to_csv(os.path.join(state.output_dir,train_matrix_file),index=None)
            test_matrix.to_csv(os.path.join(state.output_dir,test_matrix_file),index=None)
            # print(output_message)
            with open(f'{state.output_dir}/current_rule_code.json', 'w') as f:
                json.dump(code_dict, f, indent=4)  
            with open(f'{state.output_dir}/current_rule_code.txt','w') as f:
                f.write(rule_code)
            with open(f'{state.output_dir}/whole_log.txt','a') as f:
                f.write(f'Code for rules:\n{rule_code}\n---------------------------------------------------------------\n')
            with open(f'{state.output_dir}/whole_log.txt','a') as f:
                f.write(f'Current rules:\n{current_rules}\n---------------------------------------------------------------\n')
            with open(f'{state.output_dir}/whole_log.txt','a') as f:
                f.write(f'Matrix Generator Message:\n{output_matrix}\n---------------------------------------------------------------\n')
            with open(f'{state.output_dir}/{state.current_matrix}','w') as f:
                f.write(f'Matrix Generator Message:\n{output_matrix}\n---------------------------------------------------------------\n')
            current_mtx_gen +=1
            time.sleep(10)
            
            with open(f'{state.output_dir}/whole_log.txt','a') as f:
                f.write(f'Completion_tokens:\n{chat_completion.usage.completion_tokens}\nPrompt_tokens:\n{chat_completion.usage.prompt_tokens}---------------------------------------------------------------\n')    
            completion_tokens = state.completion_tokens + chat_completion.usage.completion_tokens
            prompt_tokens = state.prompt_tokens + chat_completion.usage.prompt_tokens
            
            return {'messages':[BaseMessage(content=output_matrix,sender='Matrix Generator')],'train_len':train_len,'whole_len':whole_len,'train_matrix':train_matrix_file,'test_matrix':test_matrix_file,'current_gen_count':0,'current_mtx_gen':current_mtx_gen,'completion_tokens':completion_tokens,'prompt_tokens':prompt_tokens}
        
        except TypeError or AttributeError:
            time.sleep(10)
            print('GPT RETURN ERROR, Matrix Generator')
        except Exception as e:
            print(e)
            return {'messages':[BaseMessage(content=f'**Matrix Generate Error**, please reconsider the rules.\nError:\n{e}',sender='Matrix Generator')],'current_gen_count':2,'current_mtx_gen':3}


def Coded_Matrix_Checker_o1(state:AgentState):
    '''Adjust GPT-invocation for gpt-o1'''
    GPT_model = state.GPT_model
    GPT_seed = state.GPT_seed
    # GPT_temperature = state.GPT_temperature    
    train_path = os.path.join(state.output_dir,state.train_file)
    train_set = pd.read_csv(train_path)[['name','SMILES']]#read name and SMILES
    test_path = os.path.join(state.output_dir,state.test_file)
    test_set = pd.read_csv(test_path)[['name','SMILES']]#read name and SMILES
    # whole_SMILES = pd.concat([train_set,test_set],axis=0)
    whole_identifier = pd.concat([test_set,train_set],axis=0)#!Try test set first
    whole_SMILES = list(whole_identifier['SMILES'])
    data_matrix = state.messages[-1].content
    
    
    with open(f'{state.output_dir}/current_rules.txt','r') as f:
        current_rules = f.read()
        
        
    with open(pkg_resources.resource_filename('agent.data', 'loffi.txt'),'r') as f:
        smarts_intro = f.read()
    with open(f'{state.output_dir}/current_rule_code.txt','r') as f:
        rule_code = f.read()
    system_prompt = '''
    Your task is to check whether the transformation from language rules to a numeric feature matrix is effective. The matrix is generated by a code written by Matrix Generator. If the matrix transformation is not effective, please give suggestions to improve current code.

    The feature matrix is generated with the following criteria:
    - A value of 0 if the structural description of the rule does not match the modifier.
    - A value of 1 if the structural description of the rule matches the modifier and predicts a high target value.
    - A value of -1 if the structural description of the rule matches the modifier and predicts a low target value.
    '''
    user_prompt = f'''
    !!Current Rules!!
    {current_rules}
    ------------------------------------------------------------------------------------------------
    
    !!SMILES Set!!
    {whole_SMILES}
    -------------------------------------------------------------------------------------------------
    
    !!Feature Matrix!!
    {data_matrix}
    -------------------------------------------------------------------------------------------------
    
    !!Examples for SMARTS!!
    {smarts_intro}
    -----------------------------------------------------------------------------------------------
    
    !!Current code!!
    {rule_code}
    ------------------------------------------------------------------------------------------------
    
    !!Your Target!!
    Check if the feature matrix fits the rules. Please check this carefully for each rule and each modifier. 
    Criteria:
    - Check the matrix rule by rule, SMILES by SMILES.
    - If there are many 0s in the feature matrix, verify if the rule truly does not match, especially for those 0s at the end of each rule feature list.
    - Ensure that the 1s/-1s accurately correspond to high/low target values according to each rule.
    - Take your time for thorough evaluation, but avoid making unfounded assumptions.
    
    Give suggestions for how to improve rule code write by Matrix Generator:
    - Carefully check the SMARTS write in the code based on if it describes the natural language rules correctly.
    - Give suggestions to improve the generated code
    If you find that the transformation is not successful, provide practical suggestions for your collaborator to regenerate the code generating rule matrix as SPECIFIC as possible.
    ------------------------------------------------------------------------------------------------------------------------------------------
    
    !!Format of suggestions!!
    ** Start of Suggestions **
    - ** Suggestion 1 **: ...
    - ** Suggestion 2 **: ...
    - ...
    ** End of Suggestions **
    ----------------------------------------------------------------------------------------------------------------------------------------
    
    If the transformation from language rules to numeric feature matrix is effective, add '**TRUE**' at the end of your answer. Otherwise, do not add anything at the end.
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
                f.write(f'Matrix Checker Message:\n{output_message}\n---------------------------------------------------------------\n')
                
            time.sleep(10)
            
            with open(f'{state.output_dir}/whole_log.txt','a') as f:
                f.write(f'Completion_tokens:\n{chat_completion.usage.completion_tokens}\nPrompt_tokens:\n{chat_completion.usage.prompt_tokens}---------------------------------------------------------------\n')    
            completion_tokens = state.completion_tokens + chat_completion.usage.completion_tokens
            prompt_tokens = state.prompt_tokens + chat_completion.usage.prompt_tokens
            
            return {'messages':[BaseMessage(content=output_message,sender='Matrix Checker')],'completion_tokens':completion_tokens,'prompt_tokens':prompt_tokens}
        
        except TypeError or AttributeError:
            time.sleep(100)
            print('GPT RETURN ERROR, Matrix Checker')
      

def Coded_Matrix_Generator_o1_SMARTS(state:AgentState):
    '''Try to build a totally SMARTS based rule pipeline, failed for GPT's poor understanding on SMARTS'''
    GPT_model = state.GPT_model
    GPT_seed = state.GPT_seed
    current_mtx_gen = state.current_mtx_gen
    # GPT_temperature = state.GPT_temperature
    train_path = os.path.join(state.output_dir,state.train_file)
    train_set = pd.read_csv(train_path)[['name','SMILES']]#read name and SMILES
    test_path = os.path.join(state.output_dir,state.test_file)
    test_set = pd.read_csv(test_path)[['name','SMILES']]#read name and SMILES
    # whole_SMILES = pd.concat([train_set,test_set],axis=0)
    whole_identifier = pd.concat([test_set,train_set],axis=0)#!Try test set first
    whole_SMILES = list(whole_identifier['SMILES'])
    train_len = len(train_set)
    whole_len = len(whole_SMILES)
    with open(pkg_resources.resource_filename('agent.data', 'SMARTS_semantics.txt'),'r') as f:
        smarts_semantics = f.read()
    with open(pkg_resources.resource_filename('agent.data', 'SMARTS_eg.txt'),'r') as f:
        smarts_eg = f.read()
    suggestions = '\n'
    if '**Start of Suggestions**' in state.messages[-1].content and state.messages[-1].sender == 'Matrix Checker':
        suggestions = suggestions + state.messages[-1].content
        with open(f'{state.output_dir}/current_rule_code.txt','r') as f:
            current_rule_code = f.read()
        suggestions  = suggestions +f'''----\n
        **Current code**
        {current_rule_code}'''
        
    system_prompt = '''
    **System Prompt**
    
    You are a coding assistant with expertise in RDkit. Your task is to generate Python code that takes a list of SMILES strings as input. The code should follow the provided natural language rules to convert these SMILES strings into a feature matrix using RDkit. The output matrix should be a DataFrame where each column corresponds to one rule, and each row corresponds to one SMILES string from the list. There should be number_of_SMILES rows and number_of_rules columns.
    Generate a feature matrix with the following criteria:
    - A value of 0 if the structural description of the rule does not match the SMILES.
    - A value of 1 if the structural description of the rule matches the SMILES and predicts a high target value.
    - A value of -1 if the structural description of the rule matches the SMILES and predicts a low target value.
    
    ----
    '''


    with open(f'{state.output_dir}/current_rules.txt','r') as f:
        current_rules = f.read()
        
    user_prompt = f'''
    **Examples for SMARTS**
    
    {smarts_eg}
    
    ----
    
    **Semantics of SMARTS**
    
    {smarts_semantics}
    
    ----

    **Current Rules**
    
    {current_rules}
    
    ----
    
    **Suggestions from Matrix Checker**
    
    {suggestions}
    
    ----
    
    **Target**

    Please generate Python code that follows these rules. 
    Your code should be structured in the following format:

    {{
        "prefix": "<Description of the problem and approach>",
        "imports": "<Code block containing import statements>",
        "code": "<Code block not including import statements>"
    }}

    Example:

    {{
        "prefix": "This code converts a list of SMILES strings into a feature matrix using RDkit.",
        "imports": "import pandas as pd\\nfrom rdkit import Chem\\nfrom rdkit.Chem import AllChem",
        "code": "def rule2matrix(smiles_list):\\n    rules = [ ...  \\n {{'name': 'Rule 10','patterns': ['[#7,#8,#16]',  # Multiple donor atoms (N, O, S)'[CX3](=O)[OX2H1]',  # Carboxylic acid group],'count': {{'[#7,#8,#16]': 2}},  # At least two donor atoms \\n 'exclude': ...,   \\n'prediction': 1,...
    }},
]\\n    results = []\\n  
        for smi in smiles_list:  \\n
        \\n
        results.append(row)
        df = pd.DataFrame(results, columns=[f'Rule {{i+1}}' for i in range(len(rules))])
        return df"
    }}
    
    ----
    
    Note:
    Name the function as rule2matrix, Define the function without any example to run that function.
    Using SMARTS for better substructure search.
    Treat the combination logic (separated, connected, exclude, count) of multiple substructures carefully.
    Handle possible error: when there is any error for one rule apply to one SMILES, return 0 instead.
    '''
    #! Should the example containing all(mol.HasSubstructMatch(Chem.MolFromSmarts(r)) for r in rule)?
    combined_prompt = system_prompt + user_prompt
    
    # print(system_prompt)
    # print(user_prompt)
    # while True:
    for i in range(10):
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
            code_dict = parse_llm_json(parse_LLM_json(output_message))
            code_solution = code(**code_dict)
            exec(code_solution.imports, globals())
            exec(code_solution.code, globals())
            df = rule2matrix(whole_SMILES) # type: ignore
            # rule_code = json.dumps(code_dict)
            # rule_code = output_message
            rule_code = str(code_solution.code)
            output_matrix = df.to_csv(index=False)
            # train_matrix = df.iloc[:train_len,:]
            # test_matrix = df.iloc[train_len:,:]
            #!Try test set first
            train_matrix = df.iloc[(whole_len-train_len):,:]
            test_matrix = df.iloc[:(whole_len-train_len),:]
            train_matrix_file = 'train_matrix.csv'
            test_matrix_file = 'test_matrix.csv'
            train_matrix.to_csv(os.path.join(state.output_dir,train_matrix_file),index=None)
            test_matrix.to_csv(os.path.join(state.output_dir,test_matrix_file),index=None)
            # print(output_message)
            with open(f'{state.output_dir}/current_rule_code.json', 'w') as f:
                json.dump(code_dict, f, indent=4)  
            with open(f'{state.output_dir}/current_rule_code.txt','w') as f:
                f.write(rule_code)
            with open(f'{state.output_dir}/whole_log.txt','a') as f:
                f.write(f'Code for rules:\n{rule_code}\n---------------------------------------------------------------\n')
            with open(f'{state.output_dir}/whole_log.txt','a') as f:
                f.write(f'Current rules:\n{current_rules}\n---------------------------------------------------------------\n')
            with open(f'{state.output_dir}/whole_log.txt','a') as f:
                f.write(f'Matrix Generator Message:\n{output_matrix}\n---------------------------------------------------------------\n')
            with open(f'{state.output_dir}/{state.current_matrix}','w') as f:
                f.write(f'Matrix Generator Message:\n{output_matrix}\n---------------------------------------------------------------\n')
            current_mtx_gen +=1
            time.sleep(10)
            
            return {'messages':[BaseMessage(content=output_matrix,sender='Matrix Generator')],'train_len':train_len,'whole_len':whole_len,'train_matrix':train_matrix_file,'test_matrix':test_matrix_file,'current_gen_count':0,'current_mtx_gen':current_mtx_gen}
        
        except TypeError:
            time.sleep(10)
            print('GPT RETURN ERROR, Matrix Generator')
        except Exception as e:
            print(e)
            return {'messages':[BaseMessage(content=f'**Matrix Generate Error**, please reconsider the rules.\nError:\n{e}',sender='Matrix Generator')],'current_gen_count':2,'current_mtx_gen':3}



def Coded_Matrix_Checker_o1_SMARTS(state:AgentState):
    '''Try to build a totally SMARTS based rule pipeline, failed for GPT's poor understanding on SMARTS'''
    GPT_model = state.GPT_model
    GPT_seed = state.GPT_seed
    # GPT_temperature = state.GPT_temperature    
    train_path = os.path.join(state.output_dir,state.train_file)
    train_set = pd.read_csv(train_path)[['name','SMILES']]#read name and SMILES
    test_path = os.path.join(state.output_dir,state.test_file)
    test_set = pd.read_csv(test_path)[['name','SMILES']]#read name and SMILES
    # whole_SMILES = pd.concat([train_set,test_set],axis=0)
    whole_identifier = pd.concat([test_set,train_set],axis=0)#!Try test set first
    whole_SMILES = list(whole_identifier['SMILES'])
    data_matrix = state.messages[-1].content
    
    with open(pkg_resources.resource_filename('agent.data', 'SMARTS_semantics.txt'),'r') as f:
        smarts_semantics = f.read()
    with open(pkg_resources.resource_filename('agent.data', 'SMARTS_eg.txt'),'r') as f:
        smarts_eg = f.read()
    
    with open(f'{state.output_dir}/current_rules.txt','r') as f:
        current_rules = f.read()
        
        
    # with open('/home/lnh/loffi.txt','r') as f:
    #     smarts_intro = f.read()
    with open(f'{state.output_dir}/current_rule_code.txt','r') as f:
        rule_code = f.read()
    system_prompt = '''
    **System Prompt**
    
    Your task is to check whether the transformation from language rules to a numeric feature matrix is effective. The matrix is generated by a code written by Matrix Generator. If the matrix transformation is not effective, please give suggestions to improve current code.

    The feature matrix is generated with the following criteria:
    - A value of 0 if the structural description of the rule does not match the modifier.
    - A value of 1 if the structural description of the rule matches the modifier and predicts a high target value.
    - A value of -1 if the structural description of the rule matches the modifier and predicts a low target value.
    
    ----
    '''
    
    user_prompt = f'''
    **Examples for SMARTS**
    
    {smarts_eg}
    
    ----
    
    **Semantics of SMARTS**
    
    {smarts_semantics}
    
    ----
    
    **Current Rules**
    
    {current_rules}
    
    ----
    
    **Dataset**
    
    {whole_identifier}
    
    ----
    
    **Feature Matrix**
    
    {data_matrix}
    
    ----
    
    **Current Code**
    
    {rule_code}
    
    ----
    
    **Your Target**
    Check if the feature matrix fits the rules. Please check this carefully for each rule and each modifier. 
    Criteria:
    - Check the matrix rule by rule, SMILES by SMILES.
    - If there are many 0s in the feature matrix, verify if the rule truly does not match, especially for those 0s at the end of each rule feature list.
    - Ensure that the 1s/-1s accurately correspond to high/low target values according to each rule.
    - Take your time for thorough evaluation, but avoid making unfounded assumptions.
    
    Give suggestions for how to improve rule code write by Matrix Generator:
    - Carefully check the SMARTS written in the code based on if it describes the natural language rules correctly.
    - Examine if the logic of the code matches the logic of the natural language rules.
    - Give suggestions to improve the generated code.
    If you find that the transformation is not successful, provide practical suggestions for your collaborator to regenerate the code generating rule matrix as SPECIFIC as possible.
    ----
    
    **Format of suggestions**
    
    ** Start of Suggestions **
    - ** Suggestion 1 **: ...
    - ** Suggestion 2 **: ...
    - ...
    ** End of Suggestions **
    
    ----
    
    If the transformation from language rules to numeric feature matrix is effective, conclude your answer with '**TRUE**'.
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
                f.write(f'Matrix Checker Message:\n{output_message}\n---------------------------------------------------------------\n')
                
            time.sleep(10)

            return {'messages':[BaseMessage(content=output_message,sender='Matrix Checker')]}
        
        except TypeError:
            time.sleep(10)
            print('GPT RETURN ERROR, Matrix Checker')
      
