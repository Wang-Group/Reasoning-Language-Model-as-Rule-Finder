# Reasoning Language Model as Rule Finder (RLMRF): A Case Study on Iron-Catalyzed C–H Bond Activation using 2D Metal-Organic Frameworks

- Code implementation and data availability for the essay with the same name.

- This package provides an implementation of an introspective SMARTS-based rule extraction pipeline facilitated by Large Language Model (LLM) agents with langgraph. It is designed for applications in Quantitative Structure-Activity Relationship (QSAR) studies and other experimental scenarios.

## Environment requirement
- Python >= 3.10
## Python library
- rdkit, scikit-learn, shap, openai, langgraph, dotenv, pydantic are necessary for the program. The packages can be download by conda.
```
conda install rdkit scikit-learn shap openai langgraph dotenv pydantic -c conda-forge
```
## Running the code
Codes to run the pipeline are saved in [test](test) directory
```
python test/dataset_splitting.py # generating the dictionary with the leave-one-out (LOO) datasets.
python test/main.py
```
Please set your OPENAI key and LANGCHAIN key in the GPT_agent.env.
## Components of results
[results](results) are the collection of the results and original machine learning data of the paper, containing metrics of different methods and log files of iterative pipeline of large language model (LLM).
