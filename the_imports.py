# @title IMPORTS { run: "auto" }
from sklearn.neural_network import MLPClassifier # Classifaction Model
from sklearn.neural_network import MLPRegressor # Continuous Model

# Other Models
from sklearn.linear_model import LogisticRegression # Classifaction Model
from sklearn.linear_model import LinearRegression # Continuous Model

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

from sklearn.ensemble import AdaBoostClassifier


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import RFE
from sklearn.inspection import permutation_importance

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

from sklearn.metrics import confusion_matrix

# Advance Metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import brier_score_loss
