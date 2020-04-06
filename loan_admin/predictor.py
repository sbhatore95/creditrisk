import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.ensemble import IsolationForest
import warnings
from sklearn.metrics import plot_confusion_matrix
from sklearn.decomposition import PCA
import pickle 
import codecs
from six.moves import cPickle
from abc import ABCMeta, abstractmethod
from abc import ABCMeta, abstractmethod

class Predict:
	"""Base predictor to be used for all models."""
	def __init__(self, model):
		self.model = model

	def load_model(self):
		reloaded = ""
		if(self.model == "statistical"):
			f = open('statistical.save', 'rb')
			reloaded = cPickle.load(f)
			f.close()
		elif(self.model == "ml"):
			f = open('ml.save', 'rb')
			reloaded = cPickle.load(f)
			f.close()
		elif(self.model == "statandml"):
			f = open('statandml.save', 'rb')
			reloaded = cPickle.load(f)
			f.close()
		return reloaded

	def preprocess(self, arr):
		reloaded = load_model()
		self.df = pd.DataFrame(arr.reshape(-1, len(arr)), columns = reloaded.Test_columns)
				
		for col in reloaded.le_name_mapping:
			self.df[col] = reloaded.le_name_mapping[col][self.df[col][0]]
	   
		if(len(reloaded.Nominal_Converted_features) > 0):
			z = np.zeros(len(reloaded.Nominal_Converted_features), dtype = int)
			nominal_df = pd.DataFrame(z.reshape(-1, len(z)), columns = 
				reloaded.Nominal_Converted_features)

			for col in reloaded.Nominal_Features:
				nominal_df[col+"_"+str(self.df[col][0])] = 1
				self.df = self.df.drop(col, 1)
			self.df = pd.concat([self.df, nominal_df], axis = 1)
		
		#Standardization of data frame
		
		df_std = pd.DataFrame(reloaded.std_scale.transform(self.df[reloaded.Num_Features]), 
			columns = reloaded.Num_Features)
		self.df = self.df.drop(reloaded.Num_Features, axis = 1)
		self.df = pd.concat([self.df,df_std], axis = 1)

	def predict():
		if(self.Best_Model == "KNN_model"):
			result = self.KNN_model.predict_proba(self.df)
		elif(self.Best_Model == "Logistic_model"):            
			result = self.Logistic_model.predict_proba(self.df)
		return (str(result[0][0])+ ","+str(result[0][1]))
		