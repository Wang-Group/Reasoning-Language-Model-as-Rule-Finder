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
