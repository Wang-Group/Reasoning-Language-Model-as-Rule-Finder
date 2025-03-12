Below is an overview of the workflow for performing feature selection using traditional descriptors:

1. [classification.ipynb](classification.ipynb) 
   • This notebook performs forward stepwise feature selection on two target properties: Fe/Hf and yield.  
   • The selected features and intermediate results are saved in the ***[output](output)*** directory.

2. [Fe_feature_metric.py](Fe_feature_metric.py) 
   • After the forward stepwise feature selection, this script evaluates the effectiveness of the features from the training set when targeting Fe/Hf.  
   • The evaluation results are saved in the file named ***[Fe_Hf.txt](Fe_Hf.txt)***.

3. [yield_feature_metric.py](yield_feature_metric.py)
   • Similarly, this script assesses the features on the feature set of traditional and experimental descriptors when targeting yield after the feature selection process.  
   • The results are saved in the file named ***[yield.txt](yield.txt)***.

4. [yield_noExp.txt](yield_noExp.txt)
   • Assessement of the features on the feature set of only traditional descriptors when targeting yield after the feature selection process.  

The combinations of the final selected feature set correlate to the outputs in ***[Fe_Hf.txt](Fe_Hf.txt)*** and ***[yield.txt](yield.txt)***.