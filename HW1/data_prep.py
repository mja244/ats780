# Prepare data for RF

import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#------------------Load data-------------------------

# Read csv file
path = '/Users/marcalessi/Documents/code/ats780/data'
features = pd.read_csv(path + '/combined_years_census_data_cong_dist.csv')

# Stats of each column
print(features.describe())


#-------Targets and predictors (features)------------

# predictand
predictand = np.array(features['Result'])

# Replace "DEM" with 1s and "REP" with -1s in Result column
mapping = {'REP': -1, 'DEM': 1, 'EVEN': 0}

replace_func = np.vectorize(lambda x: mapping.get(x, x)) # apply mapping to each element

predictand = replace_func(predictand)

# Remove predictand from predictors and description column
features = features.drop(['Geographic Area Name', 'Result'], axis=1)

# Save feature names
feature_list = list(features.columns)

# Convert to array
features = np.array(features)

# Replace "DEM" with 1s and "REP" with -1s in Cook PVI (baseline) column
features = replace_func(features)
print(features[:,-1])


#---------Splitting training/testing datasets---------

split_size = 0.25

train_features, test_features, train_labels, test_labels =\
	train_test_split(features, predictand, test_size=split_size, random_state=42)

np.save('npys/train_features', train_features)
np.save('npys/test_features', test_features)
np.save('npys/train_labels', train_labels)
np.save('npys/test_labels', test_labels)
np.save('npys/feature_list', feature_list)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

