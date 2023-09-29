# Functions for RF workflow

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

def visualize_tree(rf, tree_to_plot, feature_list):
	''' Visualize one tree'''
	tree = rf[tree_to_plot]
	
	export_graphviz(tree,
	        out_file='plots/rf_tree' + str(tree_to_plot) + '.dot',
	        filled=True,
	        proportion=False,
	        leaves_parallel=False,
	        class_names=['Rep', 'Dem'],
	        feature_names=feature_list)

	return tree


def calc_importances(rf, feature_list):
        '''Calculate feature importance'''
        importances = list(rf.feature_importances_)

        # List of tuples with variable and importance
        feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

        # Print out the feature and importances
        print('')
        [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
        print('')

        return importances


def plot_feat_importances(importances, feature_list):
	''' Plot the feature importance calculated by calc_importances ''' 
	plt.figure(figsize=(14, 6))

	# list of x locations for plotting
	x_values = list(range(len(importances)))

	plt.barh(x_values, importances)

	plt.yticks(x_values, feature_list)
	plt.xlabel('Importance'); plt.ylabel('Variable'); plt.title('Variable Importances')

	plt.subplots_adjust(left=0.45, right=0.95, top=0.9, bottom=0.1)

	plt.savefig('plots/feat_importance_bar_2020.png', dpi=300)
	plt.close()


def plot_perm_importance(permute, sorted_idx, feature_list):
	''' Plot the permutation importances calculated in previous cell '''
	# Sort the feature list based on 
	new_feature_list = []
	for index in sorted_idx:  
	    new_feature_list.append(feature_list[index])
	
	fig, ax = plt.subplots(figsize=(14,6))
	ax.boxplot(permute.importances[sorted_idx].T,
	       vert=False, labels=new_feature_list)
	ax.set_title("Permutation Importances")
	fig.tight_layout()

	plt.savefig('plots/perm_importance_2020.png', dpi=300)
	plt.close()


def heatmap(cm, what):
	'''Code for making heatmap for confusion matrix'''
	fig, ax = plt.subplots()
	
	cax = ax.matshow(cm, cmap='Greens')
	
	for i in range(2):
	       for j in range(2):
	               if cm[i,j] > 100:
	                       ax.text(j, i, str(cm[i,j]), va='center', ha='center', color='white')
	               else:
	                       ax.text(j, i, str(cm[i,j]), va='center', ha='center', color='k')
	
	ax.set_xticks(np.arange(2))
	ax.set_xticklabels(['Dem', 'Rep'])
	ax.set_yticks(np.arange(2))
	ax.set_yticklabels(['Dem', 'Rep'])
	
	ax.set_xlabel('Predicted', fontsize=16)
	ax.set_ylabel('Actual', fontsize=16)
	ax.xaxis.set_label_position('top')
	
	plt.savefig('plots/cm_heatmap_' + what + '_2020.png', dpi=300)
	plt.close()
