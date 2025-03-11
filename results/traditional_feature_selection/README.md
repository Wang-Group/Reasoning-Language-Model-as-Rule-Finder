# Feature selection for traditional descriptors

- [classification.ipynb](classification.ipynb): forward stepwise feature selection on Fe/Hf and yield. The results are saved in the ***[output](output)*** directory
- [Fe_feature_metric.py](Fe_feature_metric.py): After forward stepwise feature selection, the features are evaluated on corresponding train set targeted at Fe/Hf. Outputs are saved in ***[Fe_Hf.txt](Fe_Hf.txt)***.
- [yield_feature_metric.py](yield_feature_metric.py): After forward stepwise feature selection, the features are evaluated on corresponding train set targeted at yield. Outputs are saved in ***[yield.txt](yield.txt)***.