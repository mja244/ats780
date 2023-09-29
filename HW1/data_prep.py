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


#---------Splitting training/validation/testing datasets------

split_size = 0.25

trainval_features, test_features, trainval_labels, test_labels =\
	train_test_split(features, predictand, test_size=split_size, random_state=42)
train_features, val_features, train_labels, val_labels =\
	train_test_split(trainval_features, trainval_labels, test_size=0.25, random_state=42)





#-------------------Save data--------------------------
np.save('npys/train_features', train_features)
np.save('npys/test_features', test_features)
np.save('npys/val_features', val_features)
np.save('npys/features', features)

np.save('npys/train_labels', train_labels)
np.save('npys/test_labels', test_labels)
np.save('npys/val_labels', val_labels)
np.save('npys/labels', predictand)

np.save('npys/feature_list', feature_list)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Validation Features Shape:', val_features.shape)
print('Validation Labels Shape:', val_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)



#----------------------------------------------------------------
#---------------Attempt with less features-----------------------
#----------------------------------------------------------------


#------------------Load data-------------------------

# Read csv file
path = '/Users/marcalessi/Documents/code/ats780/data'
features = pd.read_csv(path + '/combined_years_census_data_cong_dist.csv')


#-------Targets and predictors (features)------------

# predictand
predictand = np.array(features['Result'])

# Replace "DEM" with 1s and "REP" with -1s in Result column
mapping = {'REP': -1, 'DEM': 1, 'EVEN': 0}

replace_func = np.vectorize(lambda x: mapping.get(x, x)) # apply mapping to each element

predictand = replace_func(predictand)

# Remove predictand from predictors and description column
features = features.drop(['Geographic Area Name', 'Result', 'Poll 1',
	'Generic Congressional Polling Average', 'Percent male', 'Percent female',
	'Median age (years)', 'Percent American Indian and Alaska Native',
	'Percent Native Hawaiian and Other Pacific Islander',
	'Percent two or more races', 'Percent high school graduate, GED, or alternative',
	"Percent Bachelor's degree or higher", 'Mean earnings (dollars)'], axis=1)

# Save feature names
feature_list = list(features.columns)

# Convert to array
features = np.array(features)

# Replace "DEM" with 1s and "REP" with -1s in Cook PVI (baseline) column
features = replace_func(features)


#---------Splitting training/validation/testing datasets------

split_size = 0.25

trainval_features, test_features, trainval_labels, test_labels =\
	train_test_split(features, predictand, test_size=split_size, random_state=42)
train_features, val_features, train_labels, val_labels =\
	train_test_split(trainval_features, trainval_labels, test_size=0.25, random_state=42)


#-------------------Save data--------------------------
np.save('npys/train_features_2', train_features)
np.save('npys/test_features_2', test_features)
np.save('npys/val_features_2', val_features)
np.save('npys/features_2', features)

np.save('npys/train_labels_2', train_labels)
np.save('npys/test_labels_2', test_labels)
np.save('npys/val_labels_2', val_labels)
np.save('npys/labels_2', predictand)

np.save('npys/feature_list_2', feature_list)



#--------------------------------------------------------------
#-----------------------2020 Midterm---------------------------
#--------------------------------------------------------------

# 2020 Election (for fun at end of project)
features_2020 = pd.read_csv(path + '/2020_census_data_cong_dist.csv')

#-------Targets and predictors (features)------------

# predictand and baseline (can we beat Cook PVI?)
predictand_2020 = np.array(features_2020['Result'])
baseline = np.array(features_2020['2019 Cook PVI'])
print(baseline)

predictand_2020 = replace_func(predictand_2020)
baseline = replace_func(baseline)

# Remove predictand from predictors and description column
features_2020 = features_2020.drop(['Geographic Area Name', 'Result'], axis=1)

# Convert to array
features_2020 = np.array(features_2020)

# Replace "DEM" with 1s and "REP" with -1s in Cook PVI (baseline) column
features_2020 = replace_func(features_2020)


#-------------------Save data--------------------------
np.save('npys/features_2020', features_2020)

np.save('npys/labels_2020', predictand_2020)

np.save('npys/baseline_2020', baseline)
