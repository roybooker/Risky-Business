# Unit-11---Risky-Business

### Mortgages, student and auto loans, and debt consolidation are just a few examples of credit and loans that people seek online. Peer-to-peer lending services such as Loans Canada and Mogo let investors loan people money without using a bank. However, because investors always want to mitigate risk, a client has asked that you help them predict credit risk with machine learning techniques.

### I have used machine learning models to predict credit risk using data you'd typically see from peer-to-peer lending services. Credit risk is an inherently imbalanced classification problem, so I have employed different techniques for training and evaluating models with imbalanced classes. I have applied imbalanced-learn and Scikit-learn libraries to build and evaluate models using the Resampling ##

--------

## Technologies

This notebook leverages Python 3.8 with the following packages in a Jupyter Notebook:
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from imblearn.metrics import classification_report_imbalanced
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import EasyEnsembleClassifier
import warnings
warnings.simplefilter(action='ignore', category=Warning)

## Installation Guide

Before running the notebook, please install the dependencies:

```python
pip install pandas
pip install sklearn
pip install matplotlib
pip install imblearn

```

## Observations

### Resampling ###

Which model had the best balanced accuracy score? 

Combination (Over and Under) Sampling using SMOTEEN algorithm balanced accuracy score of 99.9% was the highest but also SMOTE balanced accuracy score and Naive overampling balanced accuracy had scores in 99.4% mark.

Which model had the best recall score?

Combination (Over and Under) Sampling using SMOTEEN algorithm to resample had the best recall score of 1.00 for high and low risk.

Which model had the best geometric mean score?

Combination (Over and Under) Sampling using SMOTEEN algorithm to resample had the geometric mean score of 1.00 for high and low risk.

### Ensemble Learning ###

Which model had the best balanced accuracy score?

In my analysis,Easy Ensemble Classifier had the best balanced accuracy score:
     Balanced Random Forest Classifier = 0.7871246640962729
     Easy Ensemble Classifier = 0.9254565671948463
     
Which model had the best recall score?

Easy Ensemble Classifier had the best recall score of 91% for the high risk and 94% for the low risk

Which model had the best geometric mean score?

Easy Ensemble Classifier had the best geometric mean score of 93% for the high risk and 93% for the low risk

What are the top three features?

'total_rec_prncp', 'total_rec_int' and 'total_pymnt_inv' were the top three features.


### Conclusions ###
Combination (Over and Under) Sampling using SMOTEEN algorithm performed best for resampling the data, while Easy Ensemble Classifier worked best for ensemble learning of the data.

## Contributors##

By: Roy Booker

---

## License ##

MIT
