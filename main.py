import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.cluster import k_means
from sklearn.svm import SVC
from scipy.stats import pearsonr

test_data_path = 'data/test.csv'
train_data_path = 'data/train.csv'

train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)





X = test_data
Y = test_data['Survived']