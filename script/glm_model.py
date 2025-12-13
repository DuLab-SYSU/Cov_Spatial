import os
import sys
import time
import itertools
import numpy as np
import pandas as pd
import glob
from scipy import stats
import statsmodels.api as sm
#######################################################################

def Feature_selection():
	X = df[df.columns.tolist()[2:-1]]
	f_corr  = X.corr()
	feature_corr_order,corr = map(list,zip(*sorted([(col,abs(stats.pearsonr(X[col], y)[0])) for col in X.columns],key = lambda x:x[1],reverse = True)))    
	features = f_corr.columns.tolist()
	print(features)
	redund_fetures = []
	for comb in itertools.combinations(features,2):
	    if f_corr.loc[comb[0]][comb[1]] >= 0.8:
	        try:
	            redund_fetures.append(sorted(comb,key = lambda x:feature_score_order.index(x))[1])
	        except:
	            redund_fetures.append(sorted(comb,key = lambda x:feature_corr_order.index(x))[1])
	print(redund_fetures)

def GLM_model_overseas_imports():
	dominant_Omicron_sublineages = ['BA.5','BF.7','DY','XBB','EG.5', 'HK']
	for lineage in dominant_Omicron_sublineages:	
		df = pd.read_csv('../data/glm/overseas_imports/model_data_%s.csv' % lineage)
		X = sm.add_constant(df[list(df.columns)[2:-1]])
		y = df['Imports'].values
		y = np.log(y)
		gau_model = sm.GLM(y,X, family=sm.families.Gaussian())
		gau_results = gau_model.fit()
		print(gau_results.summary())
		model_result = gau_results.summary2().tables[1]
		model_result.to_csv('../results/glm/overseas_imports/model_result_%s.csv' % lineage)	

def GLM_model_domestic_exports():
	df = pd.read_csv('../data/glm/domestic_exports/model_data_domestic.csv')
	X = sm.add_constant(df[list(df.columns)[2:-1]])
	y = df['exports'].values
	gau_model = sm.GLM(y,X, family=sm.families.Gaussian())
	gau_results = gau_model.fit()
	print(gau_results.summary())
	model_result = gau_results.summary2().tables[1]
	model_result.to_csv('../results/glm/domestic_exports/model_result.csv')	
if __name__ == "__main__":
	GLM_model_overseas_imports()
	GLM_model_domestic_exports()
