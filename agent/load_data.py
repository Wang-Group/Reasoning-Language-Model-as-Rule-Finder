import os
from datetime import datetime
import json
import pkg_resources

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut

from agent.state import AgentState,BaseMessage,OverallState

def load_data(state: AgentState):
    '''load initial dataset'''
    inputs = json.loads(state.messages[-1].content.strip(),strict=True)
    target_name = inputs['target_name']
    gpt_model = inputs['GPT_model']

    split_seed = state.seed
    if target_name == 'Fe/Hf':
        append_name = 'Fe_Hf'
    elif target_name == 'modifier/SBU':
        append_name = 'modi_SBU'
        
    # generate output_folder
    date_index = datetime.now().strftime('%Y-%m-%d_%H-%M')
    date_index = append_name+'_'+date_index
    os.makedirs(date_index,exist_ok=True)
    
    # # Valid Target constraint
    valid_targets = {'Fe/Hf', 'modifier/SBU'}
    if target_name not in valid_targets:
        raise ValueError(f"Invalid target_name: {target_name}. Must be 'Fe/Hf' or 'modifier/SBU'.")
    pass

    data_path = pkg_resources.resource_filename('agent.data', 'data.csv')
    data = pd.read_csv(data_path)
    SMILES = data['SMILES']
    target_values = data[target_name]
    # mol_SBU = data['mol_SBU']
    high_or_low_loading = data[target_name] > data[target_name].median()
    # Combine SMILES and yields into a single DataFrame
    df = pd.DataFrame({
        target_name: target_values,
        # 'modifier-catalyst': mol_SBU,
        'SMILES': SMILES,
        f'{target_name}_high_or_low_value': high_or_low_loading
    })

    # Split the data into training and test sets (80% training, 20% test)
    train_set, test_set = train_test_split(df, test_size=0.20, stratify=df[f'{target_name}_high_or_low_value'], random_state=split_seed)
    train_file, test_file = 'train_set.csv', 'test_set.csv'
    train_set.to_csv(os.path.join(date_index,train_file),index=None)
    test_set.to_csv(os.path.join(date_index,test_file),index=None)
    # state.train_file = train_file
    # state.test_file = test_file
    rule_hist_path = pkg_resources.resource_filename('agent.data', f"pre_rule_metric_{append_name}.txt")
    with open(rule_hist_path,'r') as f:
        content = f.read()
    content = '< Previous Rule Metrics for Reference: >\n'+ content + '\n< Rule Metrics During the Iteration of This Program: >\n'
    with open(f'{date_index}/rule_metric_log.txt','w') as f1:
        f1.write(content)
    
    trad_hist_path = pkg_resources.resource_filename('agent.data', f"pre_trad_metric_{append_name}.txt")
    with open(trad_hist_path,'r') as f:
        content1 = f.read()
    content1 = '< Previous Accuracy for Reference: >\n'+ content1 + '\n< Accuracy and SHAP During the Iteration of This Program: >\n'
    with open(f'{date_index}/trad_metric_log.txt','w') as f1:
        f1.write(content1)  
        
    return {'messages':[BaseMessage(content=f'Data Load Successfully for {target_name}',sender='load_data')],'GPT_model':gpt_model,'target_name':target_name, 'output_dir':date_index,'train_file':train_file,'test_file':test_file,'generate_count':0}



def load_data_loo(state:OverallState):
    target_name = state['target_name']
    # Reading the dataset
    dataset_path = pkg_resources.resource_filename('agent.data', 'data.csv')
    data = pd.read_csv(dataset_path)

    # Check and create appropriate `append_name` based on `target_name`
    if target_name == 'Fe/Hf':
        append_name = 'Fe_Hf'
    elif target_name == 'modifier/SBU':
        append_name = 'modi_SBU'
    else:
        raise ValueError(f"Invalid target_name: {target_name}. Must be 'Fe/Hf' or 'modifier/SBU'.")

    # Generate output folder
    date_index = datetime.now().strftime('%Y-%m-%d_%H-%M')
    output_folder = append_name + '_' + date_index
    os.makedirs(output_folder, exist_ok=True)

    # Valid Target constraint
    valid_targets = {'Fe/Hf', 'modifier/SBU'}
    if target_name not in valid_targets:
        raise ValueError(f"Invalid target_name: {target_name}. Must be 'Fe/Hf' or 'modifier/SBU'.")

    # Prepare final dataset
    SMILES = data['SMILES']
    target_values = data[target_name]
    high_or_low_loading = data[target_name] > data[target_name].median()

    dataset = pd.DataFrame({
        target_name:target_values,
        'SMILES': SMILES,
        f'{target_name}_high_or_low_value': high_or_low_loading
    })

    rule_hist_path = pkg_resources.resource_filename('agent.data', f"pre_rule_metric_{append_name}.txt")
    with open(rule_hist_path,'r') as f:
        content = f.read()
    content = '< Previous Rule Metrics for Reference: >\n'+ content + '\n< Rule Metrics During the Iteration of This Program: >\n'
    trad_hist_path = pkg_resources.resource_filename('agent.data', f"pre_trad_metric_{append_name}.txt")
    with open(trad_hist_path,'r') as f:
        content1 = f.read()
    content1 = '< Previous Accuracy for Reference: >\n'+ content1 + '\n< Accuracy and SHAP During the Iteration of This Program: >\n'
    # Leave-One-Out cross-validation
    loo = LeaveOneOut()
    index = 1
    for train_index, test_index in loo.split(dataset):
        # Get the train and test sets
        train_set = dataset.iloc[train_index]
        test_set = dataset.iloc[test_index]
        
        # Create a new directory for each split
        split_folder = os.path.join(output_folder, str(index))
        os.makedirs(split_folder, exist_ok=True)
        
        # Save the train and test sets
        train_set.to_csv(os.path.join(split_folder, 'train_set.csv'), index=False)
        test_set.to_csv(os.path.join(split_folder, 'test_set.csv'), index=False)

        
        with open(os.path.join(split_folder,'rule_metric_log.txt'),'w') as f1:
            f1.write(content)
        
        with open(os.path.join(split_folder,'trad_metric_log.txt'),'w') as f1:
            f1.write(content1)  
        index += 1
    
    return {'output_folder':output_folder}
    