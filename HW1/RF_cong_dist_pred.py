# RF workflow for prediction of election outcome for
# each congressional district

import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score
from graphviz import Source
import pandas as pd
#import seaborn as sns
from functions import visualize_tree, calc_importances, plot_feat_importances, plot_perm_importance, heatmap
#sns.set_style('whitegrid')

#---------------------Load data----------------------------------

# Read csv file
path = '/Users/marcalessi/Documents/code/ats780/data'
features = pd.read_csv(path + '/combined_years_census_data_cong_dist.csv')

# Remove predictand from predictors and description column
features = features.drop(['Geographic Area Name', 'Result'], axis=1)
#features = features.drop(['Geographic Area Name', 'Result', 'Poll 1',
#        'Generic Congressional Polling Average', 'Percent male', 'Percent female',
#        'Median age (years)', 'Percent American Indian and Alaska Native',
#        'Percent Native Hawaiian and Other Pacific Islander',
#        'Percent two or more races', 'Percent high school graduate, GED, or alternative',
#        "Percent Bachelor's degree or higher", 'Mean earnings (dollars)'], axis=1)
feature_list = list(features.columns)

train_features = np.load('npys/train_features.npy')
test_features = np.load('npys/features_2020.npy')
val_features = np.load('npys/val_features.npy')
features_all = np.load('npys/features.npy')

train_labels = np.load('npys/train_labels.npy')
test_labels = np.load('npys/labels_2020.npy')
val_labels = np.load('npys/val_labels.npy')
labels_all = np.load('npys/labels.npy')

baseline_2020 = np.load('npys/baseline_2020.npy')


#--------------------------Train model-----------------------------
##----------------Example parameter tweaking (leaf samples)---------
#
## Tunable parameters
#
#num_trees = 24
##num_trees = np.arange(1, 31, 1)
#tree_depth = 10
##tree_depth = np.arange(1, 31)
#node_split = 2
##node_split = np.arange(2, 31)
##leaf_samples = 1
#leaf_samples = np.arange(1, 31)
#RAND_STATE = 42
#
#ps_all = []
#rs_all = []
#acc_all = []
#f1_all = []
#
#for i in leaf_samples:
#	print(i)
#	rf = RandomForestClassifier(n_estimators=num_trees,
#					random_state=RAND_STATE,
#					min_samples_split=node_split,
#					min_samples_leaf=i,
#					max_depth=tree_depth)
#	
#	# Train the model
#	
#	rf.fit(train_features, train_labels)
#	
#	
#	#--------------------------Make predictions-------------------------
#	
#	predictions = rf.predict(val_features)
#	
#	ps = precision_score(val_labels, predictions)
#	rs = recall_score(val_labels, predictions)
#	acc = accuracy_score(val_labels, predictions)
#	f1 = f1_score(val_labels, predictions)
#	ps_all.append(ps)
#	rs_all.append(rs)
#	acc_all.append(acc)
#	f1_all.append(f1)
#	
#
##------------------plot scores--------------------------------
#
#fig, ax = plt.subplots()
#
#ax.plot(np.arange(1, 31, 1), ps_all, label='Precision')
#ax.plot(np.arange(1, 31, 1), rs_all, label='Recall')
#ax.plot(np.arange(1, 31, 1), acc_all, label='Accuracy')
#ax.plot(np.arange(1, 31, 1), f1_all, label='f1')
#ax.set_xlabel('Leaf samples')
#ax.set_ylabel('Score')
#
#ax.legend()
#
#plt.savefig('plots/leaf_samples_metrics.png', dpi=300)
#plt.close()


#-------------Final model--------------------------------------

num_trees = 24
tree_depth = 10
node_split = 2
leaf_samples = 1
RAND_STATE = 42

rf = RandomForestClassifier(n_estimators=num_trees,
                                random_state=RAND_STATE,
                                min_samples_split=node_split,
                                min_samples_leaf=leaf_samples,
                                max_depth=tree_depth)

# Train the model

rf.fit(train_features, train_labels)

predictions = rf.predict(val_features)


#------------Explainability of Tree---------------------------

## visualize confusion matrix

# map numbers to REP and DEM (this is not necessary, but just easier to read)
mapping = {-1: 'REP', 1: 'DEM', 0: 'EVEN'}

replace_func = np.vectorize(lambda x: mapping.get(x, x))
val_labels = replace_func(val_labels)
predictions = replace_func(predictions)

cm = confusion_matrix(val_labels, predictions)

# plot heatmap
heatmap(cm, 'val')


## visualize one tree

visualize_tree(rf, 4, feature_list)


## Feature importance

importances = calc_importances(rf, feature_list)

plot_feat_importances(importances, feature_list)


## Permutation importance

permute = permutation_importance(
	rf, features_all, labels_all, n_repeats=20, random_state=RAND_STATE)

sorted_idx = permute.importances_mean.argsort()

plot_perm_importance(permute, sorted_idx, feature_list)


#------------------Apply to testing data----------------------
predictions_test = rf.predict(test_features)
###
base_anom = baseline_2020 - test_labels
print(base_anom)

pred_anom = predictions_test - test_labels
print(pred_anom)
###
test_labels = replace_func(test_labels)
predictions_test = replace_func(predictions_test)

test_labels_ = test_labels[baseline_2020 != 0]
predictions_test_ = predictions_test[baseline_2020 != 0]

cm_test = confusion_matrix(test_labels_, predictions_test_)

# plot heatmap
heatmap(cm_test, 'test')


# 2020 testing
baseline_2020_ = baseline_2020[baseline_2020 != 0]

baseline_2020_ = replace_func(baseline_2020_)
cm_baseline = confusion_matrix(test_labels_, baseline_2020_)
heatmap(cm_baseline, 'baseline')

## where did cook pvi get wrong and model got right?
#
#print(baseline_2020)
#print(test_labels)
