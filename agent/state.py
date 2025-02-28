import operator
from typing_extensions import TypedDict
from typing import Sequence, List, Annotated,Literal

from pydantic import BaseModel, Field

class BaseMessage(BaseModel):
    content: str
    sender: str

class AgentState(BaseModel):
    #Define State with default settings
    messages: Annotated[Sequence[BaseMessage], operator.add]
    reaction_background: str = Field(default='background')
    target_name: str = Literal['Fe/Hf','modifier/SBU','yield']
    Fe_pred_flag: bool = Field(default=False)
    
    #LOO utilities
    train_performance: list = Field(default=[]) # 5 Fold accuracy on validation set
    test_prediction: list = Field(default=[])
    current_val_performance: float = Field(default=0.0)
    # loo_log: list = Field(default=[])
    
    output_dir: str = Field(default='instant')
    train_file: str = Field(default='train_set.csv')
    test_file: str = Field(default='test_set.csv')
    train_matrix: str = Field(default='train_matrix.csv')
    test_matrix: str = Field(default='test_matrix.csv')
    exp_train: str = Field(default='exp_train.csv')
    exp_test: str = Field(default='exp_test.csv')
    selected_train_matrix: str = Field(default='selected_train_mtx.csv')
    selected_test_matrix: str = Field(default='selected_test_mtx.csv')
    current_matrix: str = Field(default='current_matrix.txt')
    generate_count: int = Field(default=0)
    current_gen_count: int = Field(default=0)
    current_mtx_gen: int = Field(default=0)
    train_len: int = Field(default=0)
    whole_len: int = Field(default=0)
    # GPT args for generator
    Tran_model: str = Literal['ETC', 'RFC'] # ExtraTreesClassifier or randomForestClassifier
    GPT_model: str = Literal['gpt-4o-2024-08-06', 'gpt-4o-mini-2024-07-18','deepseek-r1']
    GPT_temperature: float = Field(default=0.3)#!Use different temperature and top_p for different chatbot  GPT: 0.5/0.5, 0.3/0.2 and 0.2/0.1
    GPT_seed: int = Field(default=42)
    seed: int = Field(default=42) #other random seeds
    
    completion_tokens: int = Field(default=0)
    prompt_tokens: int = Field(default=0)
    
class OverallState(TypedDict):
    output_folder: str
    GPT_model: str = Literal['gpt-4o-2024-08-06', 'gpt-4o-mini-2024-07-18','deepseek-r1']
    target_name: str = Literal['Fe/Hf','modifier/SBU','yield']
    test_true: Annotated[list, operator.add]
    test_pred: Annotated[list, operator.add]
    loo_log: Annotated[list,operator.add]
    test_accuracy: float
    
    
    