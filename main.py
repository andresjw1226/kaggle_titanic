import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.cluster import k_means
from sklearn.svm import SVC
from scipy.stats import pearsonr
from sqlalchemy import column

test_data_path = 'data/test.csv'
train_data_path = 'data/train.csv'

train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

# Remove irrelevant columns
columns = list(train_data.columns)
source_columns = columns
source_columns.remove('Ticker')
source_columns.remove('Cabin')

# Remove row includes missing values
df = train_data[source_columns]
clean_df = df.dropna()

# Filter by departing locations
c = clean_df[clean_df['Embarked'] == 'C']
s = clean_df[clean_df['Embarked'] == 'S']
q = clean_df[clean_df['Embarked'] == 'Q']

# Check wether there is high correlation between'Fare' and 'Pclass' 
corr_c, _c = pearsonr(c['Pclass'], c['Fare']) 
corr_s, _s = pearsonr(s['Pclass'], s['Fare']) 
corr_q, _q = pearsonr(q['Pclass'], q['Fare']) 




X = test_data
Y = test_data['Survived']