## Reasoning Language Model as Rule Finder (RLMRF): A Case Study on Iron-Catalyzed C–H Bond Activation using 2D Metal-Organic Frameworks

- This directory containing the code implementation and data availability for the essay with the same name.

- This package([agent](agent)) provides an implementation of an introspective SMARTS-based rule extraction pipeline facilitated by Large Language Model (LLM) agents with langgraph. It is designed for applications in Quantitative Structure-Activity Relationship (QSAR) studies and other experimental scenarios.

### Prerequisites
- **Python Version:** Ensure you have Python 3.10 or above.

### Python Libraries
The following libraries are required:
- `rdkit`
- `scikit-learn`
- `shap`
- `openai`
- `langgraph`
- `dotenv`
- `pydantic`

You can install these dependencies using Conda:
```bash
conda install rdkit scikit-learn shap openai langgraph dotenv pydantic -c conda-forge
```

### Setting Up the Environment
1. **API Key Setup:**
   - You need to have API keys for OpenAI and Langchain.
   - Create a file named "*[GPT_agent.env](GPT_agent.env)*" and set your API keys:
     ```
     OPENAI_API_KEY=your_openai_key
     LANGCHAIN_API_KEY=your_langchain_key
     ```

### Running the Project
The implementation includes a predefined pipeline with sample test cases:
1. **Data Preparation:**
   Execute the script to generate datasets with leave-one-out (LOO) cross-validation:
   ```bash
   python test/dataset_splitting.py
   ```
   This prepares the dataset dictionaries necessary for the LOO evaluation.
   
2. **Main Pipeline Execution:**
   Run the main pipeline script:
   ```bash
   python test/main.py
   ```
   This script facilitates the application of the introspective rule extraction methodology powered by LLM.

### Results
- **Location:** The results, including original machine learning data, metrics of various approaches, and logs of the LLM iterative process, are saved in the *[results](results)* directory.

This setup not only supports QSAR studies but can be adapted for any experimental scenario involving rule extraction with machine learning models. If you have additional questions or need further assistance, feel free to ask!
