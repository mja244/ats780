# RF workflow for prediction of election outcome for
# each congressional district

import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
from sklearn.inspection import permutation_importance

#---------------------Load data----------------------------------

# Read csv file
path = '/Users/marcalessi/Documents/code/ats780/data'
features = pd.read_csv(path + '/combined_years_census_data_cong_dist.csv')

# Remove predictand from predictors and description column
features = features.drop(['Geographic Area Name', 'Result'], axis=1)
feature_list = list(features.columns)

train_features = np.load('npys/train_features.npy')
test_features = np.load('npys/test_features.npy')
train_labels = np.load('npys/train_labels.npy')
test_labels = np.load('npys/test_labels.npy')


#----------------------Establish baseline-------------------------

# The baseline predictions here are the 2019 Cook Paristan Voting
# Index (Cook PVI) values (here set as DEM=1 or REP=-1)

baseline_preds = test_features[:, feature_list.index('2019 Cook PVI')]

# Baseline errors (mean absolute error)
mae_baseline_errors = abs(baseline_preds - test_labels)
print('Baseline error (MAE): ', round(np.mean(mae_baseline_errors), 2))


#--------------------------Train model-----------------------------

# Tunable parameters

num_trees = np.arange(2, 50, 2)
tree_depth = None
node_split = 2
leaf_samples = 1
RAND_STATE = 42

error = []

for i in num_trees:
	print(i)
	rf = RandomForestRegressor(n_estimators = int(i),
					random_state = RAND_STATE,
					min_samples_split = node_split,
					min_samples_leaf = leaf_samples,
					max_depth = tree_depth)
	
	# Train the model
	
	rf.fit(train_features, train_labels)
	
	
	#--------------------------Make predictions-------------------------
	
	predictions = rf.predict(test_features)
	
	# What does the MAE look like for our RF model?
	mae_errors = abs(predictions - test_labels)
	
	print('Error (MAE): ', round(np.mean(mae_errors), 2))
	error.append(round(np.mean(mae_errors), 2))

print(error)

fig, ax = plt.subplots()

ax.plot(np.arange(2, 50, 2), error)

plt.savefig('plots/num_trees_mae.png', dpi=300)
