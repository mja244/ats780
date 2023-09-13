# Prepare data for RF

import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
from sklearn.inspection import permutation_importance

#------------------Load data-------------------------

# Read csv file
path = '/Users/marcalessi/Documents/code/ats780/data'
features = pd.read_csv(path + '/combined_years_census_data_cong_dist.csv')

# Stats of each column
print(features.describe())


#-------Targets and predictors (features)------------

# predictand
predictand = np.array(features['Result'])

# Remove predictand from predictors and description column
features = features.drop(['Geographic Area Name', 'Result'], axis=1)

# Save feature names
feature_list = list(features.columns)

# Convert to array
features = np.array(features)


#---------Splitting training/testing datasets---------

split_size = 0.25

train_features, test_features, train_labels, test_labels =\
	train_test_split(features, predictand, test_size=split_size, random_state=42)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)
