import os
import time
import datetime

import numpy as np
import pandas as pd
import pkg_resources

from agent.state import AgentState,BaseMessage
from agent.client import client

def read_last_lines_as_string(file_path, n=80):
    '''Read the last 80 lines in the rule discussion'''
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        last_lines = lines[-n:] if len(lines) >= n else lines
        return '\n'.join(last_lines)
    
def Rule_Generator(state:AgentState):
    '''Generate rules based on the SMILES, name corresponding to the target value of each molecule.'''
    
    current_gen_count = state.current_gen_count # Count the generation number of the linguistic rule generation part of this iteration
    generate_count = state.generate_count # Count the total number of GPT rule generation
    GPT_model = state.GPT_model # set the GPT model
    GPT_seed = state.GPT_seed #set the seed for GPT
    # GPT_temperature = state.GPT_temperature
    train_path = os.path.join(state.output_dir,state.train_file)
    train_set = pd.read_csv(train_path).iloc[:,1:] # skip original regression target value column
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
    ----------------------------------------------------------------------------------------------------------------------
    
    !!Format of Rules!!
    **Start of Rules
    - **Rule 1**:...
    - **Rule 2**:...
    - ...
    **End of Rules
    -----------------------------------------------------------------------------------------------------------------------\n
    !!Current Rules and corresponding Advice!!
    {current_rules}
    '''
    add_prompt = '''\n-------------------------------------------------------------------------------------------------------\nPlease generate rules! You're suggested to find new rules and abandon those bad rules in current rules'''
    
    rule_advice = ''
    last_msg = state.messages[-1].content
    if last_msg.find("** Start of Advice **") != -1:
        rule_advice = state.messages[-1].content

    if rule_advice != '':
        add_prompt = rule_advice + add_prompt
    
    user_prompt = user_prompt + add_prompt
        
    # while True:
    for i in range(100):
        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                model=GPT_model,
                temperature = 0.5,
                seed = GPT_seed,
                top_p=0.5,
                n=1
            )# more creative when generating rules

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
            return {'messages':[BaseMessage(content=output_message,sender='Rule Generator')],'reaction_background':reaction_background,'generate_count':generate_count,'current_gen_count':current_gen_count}
        
        except TypeError:
            time.sleep(100)
            print('GPT RETURN ERROR, Rule Generator')
    
    
def Rule_Commenter(state:AgentState):
    '''give comment on GPT generated rules based on Clarity, Property Insight, Complexity, Coverage and Balance.'''
    GPT_model = state.GPT_model
    GPT_seed = state.GPT_seed
    # GPT_temperature = state.GPT_temperature
    train_path = os.path.join(state.output_dir,state.train_file)
    train_set = pd.read_csv(train_path).iloc[:,1:]#skip original regression target value column
    current_rules = state.messages[-1].content
    reaction_background = state.reaction_background
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
                f.write(f'Rule Commenter Message:\n{output_message}\n---------------------------------------------------------------\n')
            with open(f'{state.output_dir}/Rule_discussion_log.txt','a') as f:
                f.write(f'Rule Commenter Message:\n{output_message}\n---------------------------------------------------------------\n')  
            
            time.sleep(10)

            return {'messages':[BaseMessage(content=output_message,sender='Rule Commenter')]}
        
        except TypeError:
            time.sleep(100)
            print('GPT RETURN ERROR, Rule Commenter')
    
def Rule_Advisor(state:AgentState):
    '''give advice to improve current rules based on the message input'''
    GPT_model = state.GPT_model
    GPT_seed = state.GPT_seed
    # GPT_temperature = state.GPT_temperature
    train_path = os.path.join(state.output_dir,state.train_file)
    train_set = pd.read_csv(train_path).iloc[:,1:]#skip original regression target value column
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
        input_discussion = '''It's difficult to generate numeric feature matrix from current rules'''
        # input_discussion = state.messages[-1].content
    elif state.messages[-1].sender == 'Tradition Calculator':
        #Model training error because of almost all 0 matrix
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

    Your advice must be directly practical on how to improve the SMILES based rules.
    -------------------------------------------------------------------------------------------------

    If you believe the rules are good enough in this stage, conclude your answer with "**TRUE**". If not, DO NOT add anything at the end.
    Note that advice from Project Manager is important. That means current rule need to be optimized.
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
                f.write(f'Rule Advisor Message:\n{output_message}\n---------------------------------------------------------------\n')
                
            time.sleep(10)
            return {'messages':[BaseMessage(content=output_message,sender='Rule Advisor')]}
        
        except TypeError:
            time.sleep(10)
            print('GPT RETURN ERROR, Rule Advisor')
            
            
            


def Rule_Generator_o1_SMARTS(state:AgentState):
    '''Try to build a totally SMARTS based rule pipeline, failed for GPT's poor understanding on SMARTS'''
    current_gen_count = state.current_gen_count
    generate_count = state.generate_count
    GPT_model = state.GPT_model
    GPT_seed = state.GPT_seed
    # GPT_temperature = state.GPT_temperature
    train_path = os.path.join(state.output_dir,state.train_file)
    train_set = pd.read_csv(train_path).iloc[:,1:]#skip original regression target value column
    
    if os.path.exists(f'{state.output_dir}/current_rules.txt'):
        with open(f'{state.output_dir}/current_rules.txt','r') as f:
            current_rules = f.read()
        current_rules = '\n **Current Rules**\n There may be advise based on imporvement of current rules, here are current rules:\n'+current_rules
    else:
        current_rules = ''
        
    SMARTS_path = pkg_resources.resource_filename('agent.data', 'SMARTS_intro.txt')
    with open(SMARTS_path,'r') as f:
        smarts_guide = f.read()
    reaction_background = '''
    In the pursuit of efficient methodologies for the selective functionalization of aliphatic alcohols, the remote δ-C(sp³)–H bond activation has emerged as a promising strategy. Our study focuses on a radical-mediated functionalization process where di-tert-butyl azodicarboxylate (DBAD) is utilized as a substrate for δ-amino alcohol synthesis, catalyzed by Fe(OTf)₃ under mild conditions. The reaction occurs in the presence of tetrabutylammonium chloride (TBACl) in acetonitrile, activated by 390 nm LEDs.

    Crucial to this approach is the use of a Metal-Organic Layer (Hf-TPY-MOL), composed of hafnium-oxygen clusters and terpyridine ligands, which stabilizes and enhances the activity of the iron catalyst. By modifying the secondary building unit (SBU) of Hf-TPY-MOL with various carboxylates, the reaction’s yield and selectivity can be tuned. Our work investigates this system, using 1-pentanol and DBAD, to develop a selective pathway to δ-amino alcohols through radical-based transformations.

    **Understanding the Core Structure of Hf-TPY-MOL**

    - **Hf-TPY-MOL** is a Metal-Organic Layer (MOL) based on hafnium-oxygen clusters (SBUs, secondary building units) and terpyridine (TPY) ligands.
    - **Hf to Oxygen Bonds:**: Hf-O (μ3-O, μ3-OH): ~2.268 Å (can vary slightly depending on coordination environment)
    - **Size on Hf cluster**
    - Given that the two carboxylates coordinate on opposite sides of the Hf-O cluster, the distance between the two carboxylate carbon atoms is ~8.541 Å
    - **Size of TPY ligand**
    -Considering an axis passing through one carboxylate and the centers of two aromatic rings and the nitrogen atom, the distance from the carboxylate carbon to the nitrogen atom is ~8.660 Å
    - The **SBU (Hf₆(μ₃-O)₄(μ₃-OH)₄)** is the core unit, which contains six hafnium atoms connected through oxo (O²⁻) and hydroxo (OH⁻) bridges. This creates a robust, highly stable 3D architecture.
    - The **TPY ligand**, specifically **4'-(4-benzoate)[2,2';6',2''-terpyridine]-5,5''-dicarboxylate**, has carboxylic groups that coordinate with the Hf-O cluster, and the three nitrogen atoms in the TPY introduce sites where the Fe catalyst can be anchored.
    - The Hf-O cluster has an octahedral geometry, where each face contains an oxygen atom and each edge is coordinated by a carboxylate group. Modifying the Hf-TPY-MOL involves replacing some of the original carboxylate groups with other molecules, thereby functionalizing the material.
    
    **Existing Knowledge**

    - The reaction takes place in acetonitrile as the solvent.
    - Catalyst: Fe³⁺ loaded on Hf-TPY-MOL acts as a heterogeneous catalyst.
    - The δ-selectivity vanishes if there is no –OH group in the reactant molecule, indicating that the –OH group first changes into an –O· radical, and a six-membered ring transition state occurs during the hydrogen atom transfer (HAT) process to generate the δ-carbon radical.
    - If there is no TBACl, the reaction yield is zero, indicating the significant impact of chloride ions on the electron transition.
    - **Mechanism**: After the ligand-to-metal charge transfer (LMCT) reduces Fe³⁺ to Fe²⁺, a carbon radical is generated. This carbon radical adds to the DBAD to generate a nitrogen radical, which undergoes a single-electron transfer (SET) process (oxidizing Fe²⁺ back to Fe³⁺) to produce the product and complete the catalytic cycle.
    
    **Understanding Target Value**
    
    The primary goal is to optimize and control the yield of the remote δ-C(sp3)–H bond functionalization reaction. It has been observed that the modifier loading on the catalyst (modifier/SBU), the fraction of Fe to Hf in the catalyst (Fe/Hf), and the total loading of Fe (Fe_loading) significantly impact the yield. It's assumed that modifier/SBU, Fe/Hf, and yield These parameters are influenced by different types of molecular modifiers.
    
    ---
   '''
    
    system_prompt = f'''
    **System Prompt**

    You are collaborating with other agents on a research program focused on the problem described below. Utilizing your chemical insights, please analyze the dataset provided and generate rules that describe the relationship between molecular modifiers (depicted as SMILES strings) and the relative high or low value of `{state.target_name}`.

    **Note**: If any advice is given, take it into consideration.
    
    ----
    
    '''
    user_prompt = f'''
    **Reaction Background**
    
    {reaction_background}
    
    ---
    
    **DataSet**
    
    {train_set}
    
    ---
    
    **Requirements for Generated Rules**

    1. Imagine the structure of Hf-TPY-MOL, considering how molecular polarizabilities, dipoles, (induced) dipole interactions, π-π interactions, steric properties, hydrophobic properties, and solvent effects affect the photocatalytic process.
    2. Consider how the electron transition process is affected.
    3. Knowledge of electronegativity, Hard and Soft Acid and Base theory, the spectrochemical series, coordination field theory, the trans effect, electron donor/withdraw properties, coordinating properties, and chemical bond lengths may be helpful.
    4. Rules should illustrate direct combinations of sub-structures (function groups) features in the modifiers. Combining multiple sub-structures (function groups) is recommended.
    5. For rules describe general physical-chemical properties, the combination logic (separated, connected, exclude, count) of multiple substructures are suggested. For example: Electron-Withdrawing Groups (EWGs) contaiing Carbonyl Group, Nitro Group, Halogens and Trifluoromethyl Group.
    6. When generating rules, keep the underlying knowledge and properties mentioned above in mind.
    7. Each rule should clearly predict whether the target value is high or low for any SMILES structure that fits its description.
    8. Prioritize rules that cover a broader range of the dataset.
    9. Generate between 5 and 15 rules.
    10. Maintain a suitable balance between general rules with higher coverage and complex rules with lower coverage.
    
    ---

    **Format of Rules**

    **Start of Rules**

    - **Rule 1**: ...
    - **Rule 2**: ...
    - ...

    **End of Rules**
    
    ----
    
    {current_rules}
    '''
    add_prompt = '''\n-------------------------------------------------------------------------------------------------------\nPlease generate rules! You are suggested to polish the current rule set. If there are any meaningless rule in current rules, replace it by generating a new rule.'''
    
    rule_advice = ''
    last_msg = state.messages[-1].content
    if last_msg.find("** Start of Advice **") != -1 and state.messages[-1].sender == 'Rule Advisor':
        rule_advice = state.messages[-1].content

    if rule_advice != '':
        #!!format of rule_advice
        add_prompt = rule_advice+ add_prompt
    combined_prompt = system_prompt + user_prompt
        
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
            generate_count += 1
            current_gen_count += 1
            output_message = chat_completion.choices[0].message.content.strip()
            with open(f'{state.output_dir}/whole_log.txt','a') as f:
                f.write(f'Rule Generator Message:\n{output_message}\n---------------------------------------------------------------\n')
            with open(f'{state.output_dir}/Rule_discussion_log.txt','a') as f:
                f.write(f'Rule Generator Message:\n{output_message}\n---------------------------------------------------------------\n')    
            with open(f'{state.output_dir}/current_rules.txt','w') as f:
                f.write(f'{output_message}---------------------------------------------------------------\n')
            # output_message  = extract_rules(output_message)
            time.sleep(10)
            return {'messages':[BaseMessage(content=output_message,sender='Rule Generator')],'reaction_background':reaction_background,'generate_count':generate_count,'current_gen_count':current_gen_count}
        
        except TypeError:
            time.sleep(10)
            print('GPT RETURN ERROR, Rule Generator')
            
            
def Rule_Commenter_o1_SMARTS(state:AgentState):
    '''Try to build a totally SMARTS based rule pipeline, failed for GPT's poor understanding on SMARTS'''
    GPT_model = state.GPT_model
    GPT_seed = state.GPT_seed
    # GPT_temperature = state.GPT_temperature
    train_path = os.path.join(state.output_dir,state.train_file)
    train_set = pd.read_csv(train_path).iloc[:,1:]#skip value column
    current_rules = state.messages[-1].content
    with open(pkg_resources.resource_filename('agent.data', 'SMARTS_semantics.txt'),'r') as f:
        smarts_semantics = f.read()
    reaction_background = state.reaction_background
    system_prompt = '''
    **System Prompt**
    
    You are collaborating with other agents on a research program focused on the a catalytical problem. 
    Your target is give comment to rules for advance rules and judge if the rules is enough for latter tasks.
    
    ----
    
    '''
    user_prompt=f'''
    **Reaction Background**
    
    {reaction_background}
    
    ----
    
    *Current Rules**
    
    {current_rules}
    
    ----
    
    **DataSet**
    
    {train_set}
    
    ----
    
    **Scoring Criteria for Rules**
    
    # Importance in descending order:
    1. **Clarity**: Determine if you can clearly tell whether the target value is high or low when a modifier matches the structural description of the rule.
    2. **Property Insight**: Assess whether there is adequate physical-chemical insight corresponding to the properties of the modifier and the reaction. Imagine the structure of Hf-TPY-MOL, considering how molecular polarizabilities, dipoles, (induced) dipole interactions, π-π interactions, steric properties, hydrophobic properties, and solvent effects affect the photocatalytic process. Knowledge of electronegativity, Hard and Soft Acid and Base theory, the spectrochemical series, coordination field theory, the trans effect, electron donor/withdraw properties, coordinating properties, and chemical bond lengths may be helpful.
    3. **Complexity**: Ensure the rules are elaborate enough: the flexible logic (separated, connected, exclude, count) of multiple substructures is used for the rules.
    4. **Coverage**: Verify that at least 2 data points support a rule. More support corresponds to a higher score.
    5. **Balance**: Consider the balance between complexity and coverage. Although complex rules with detailed sub-structures are valuable, sometimes simpler rules with higher coverage are also effective. Try to achieve a balanced approach.
    
    ----
    
    **Comment Format**
    
    **Start of Comments
    - **Comment 1**:...
    - **Comment 2**:...
    - ...
    **End of Comments
    
    ----
    
    **Your Target**
    
    Scan the rules one by one and score them based on the following criteria. Sum the scores for each rule and provide your comments on the effectiveness of current rules.
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

            return {'messages':[BaseMessage(content=output_message,sender='Rule Commenter')]}
        
        except TypeError:
            time.sleep(10)
            print('GPT RETURN ERROR, Rule Commenter')
    
def Rule_Advisor_o1_SMARTS(state:AgentState):
    '''Try to build a totally SMARTS based rule pipeline, failed for GPT's poor understanding on SMARTS'''
    GPT_model = state.GPT_model
    GPT_seed = state.GPT_seed
    # GPT_temperature = state.GPT_temperature
    train_path = os.path.join(state.output_dir,state.train_file)
    train_set = pd.read_csv(train_path).iloc[:,1:]#skip value column
    with open(pkg_resources.resource_filename('agent.data', 'SMARTS_semantics.txt'),'r') as f:
        smarts_semantics = f.read()
    with open(f'{state.output_dir}/current_rules.txt','r') as f:
        current_rules = f.read()
    reaction_background = state.reaction_background
    # reaction_background = '''
    # You are tasked with solving the following problem: a radical-mediated remote δ-C(sp3)–H bond functionalization reaction of aliphatic alcohols using di-tert-butyl azodicarboxylate (DBAD) as the substrate. The reaction is catalyzed by FeCl3 in the presence of tetrabutylammonium chloride (TBACl) and conducted in acetonitrile solvent under irradiation with 390 nm light-emitting diodes (LEDs).

    # In addition, the reaction employs Hf-TPY-MOL, a Metal Organic Layer composed of hafnium-oxygen clusters (SBU, Secondary Building Unit) coordinated with terpyridine ligands. This setup is used to capture and stabilize the Fe ion. The SBU of the MOL can be modified using a molecular modifier to affect the reactivity of the catalyst Hf-TPY-MOL(Fe).

    # The primary goal is to optimize and control the yield of the remote δ-C(sp3)–H bond functionalization reaction. It has been observed that the modifier loading on the catalyst (modifier/SBU), the fraction of Fe to Hf in the catalyst (Fe/Hf), and the total loading of Fe (Fe_loading) significantly impact the yield. It's assumed that modifier/SBU, Fe/Hf, and yield These parameters are influenced by different types of molecular modifiers.'''
    
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
        input_discussion = '''It's difficult to generate numeric feature matrix from current rules'''
        # input_discussion = state.messages[-1].content
    elif state.messages[-1].sender == 'Tradition Calculator':
        input_discussion = state.messages[-1].content
    system_prompt='''
    **System Prompt**
    
    You are tasked with reading the discussion on the current rules. 
    Based on the reaction data, provide practical advice for the rule generator to create improved rules. 
    
    ----
    '''
    user_prompt=f'''
    **Reaction Background**
    
    {reaction_background}
    
    ----
    
    **Current Rules**
    
    {current_rules}
    
    ----
    
    **DataSet**
    
    {train_set}
    
    ----
    
    **Discussions**
    
    {input_discussion}
    
    ----
    
    **Advice Format**
    
    ** Start of Advice **
    - ** Advice 1 **: ...
    - ** Advice 2 **: ...
    - ...
    ** End of Advice **
    
    ----
    
    **Your Target**
    
    Your advice must be directly practical on how to improve the rules set or directly give new rules.
    If you believe the rules are good enough in this stage, conclude your answer with "**TRUE**". If not, DO NOT add anything at the end.
    Note that suggestions from Project Manager are important, take them on priority. That means current rule need to be optimized.
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
            return {'messages':[BaseMessage(content=output_message,sender='Rule Advisor')]}
        
        except TypeError:
            time.sleep(10)
            print('GPT RETURN ERROR, Rule Advisor')
            
            
            
def Rule_Generator_o1(state:AgentState):
    '''Adjust GPT-invocation for gpt-o1'''
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
    '''Adjust GPT-invocation for gpt-o1''' 
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
    '''Adjust GPT-invocation for gpt-o1'''
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
    elif state.messages[-1].sender == 'Tradition Calculator':
        #Model training error because of almost all 0 matrix
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