import abc
from .models import SavedState
from loan_admin.models import UploadFile, Criteria, CriteriaHelper, Configuration, Feature
from .project import *
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
#import xgboost as xgb
from sklearn import svm
from sklearn.ensemble import IsolationForest
import warnings
from sklearn.metrics import plot_confusion_matrix
from sklearn.decomposition import PCA
import pickle 
import codecs
from six.moves import cPickle
import os
from django.conf import settings

class RuleBasedStrategyAbstract(object):
	__metaclass__ = abc.ABCMeta

	@abc.abstractmethod
	def get_result(self, loan_id):
		"""Required Method"""

class RuleBasedStrategy(RuleBasedStrategyAbstract):
	def get_result(self, loan_id):
		f = open(os.path.join(settings.MEDIA_ROOT, 'credit_risk/dataset/test_id_dataset.csv'), 'r')
		line = f.readline()
		array = line.split(',')
		array[-1] = array[-1].replace('\n', '')
		Dict = {}
		for i in range(0, len(array)):
			Dict[array[i]] = i
		line = f.readline()
		while line != "":
			sp = line.split(',')
			if(sp[0] == loan_id):
				break
			line = f.readline()
		sp[-1] = sp[-1].replace('\n', '')
		cri = Criteria.objects.all()
		conf = Configuration.objects.all()
		ans = 0
		for configuration in conf:
			for criteria in cri:
				helper = CriteriaHelper.objects.all().filter(criteria=criteria)
				truth = (criteria.feature == configuration.feature and 
					criteria.product == configuration.product and 
					criteria.category == configuration.category)
				if(truth):
					print("------------------")
					ins = Feature.objects.filter(name=criteria.feature).first()
					if(ins):
						if(ins.status == False):
							continue
					print(criteria.feature)
					# if(ins.status == False):
					# 	continue
					if(criteria.data_source == '3'): # for SQL
						feature = criteria.api.split(' ')[1]
					if(feature in array):
						feature_score = 0
						for crhelper in helper:
							# For categorical
							tup = crhelper.entry.split(' ')
							if(tup[0] == "is"):
								if(tup[1] == sp[Dict[feature]]):
									feature_score += crhelper.score
							elif(tup[0] == ">"):
								if(int(sp[Dict[feature]]) > int(tup[1])):
									feature_score += crhelper.score
							elif(tup[0] == "<"):
								if(int(sp[Dict[feature]]) < int(tup[1])):
									feature_score += crhelper.score
							elif(tup[0] == ">="):
								if(int(sp[Dict[feature]]) >= int(tup[1])):
									feature_score += crhelper.score
							elif(tup[0] == "<="):
								if(int(sp[Dict[feature]]) <= int(tup[1])):
									feature_score += crhelper.score
						ans += configuration.weightage * feature_score
		f.close()
		return str(ans)

class DataBasedStrategyAbstract(object):
	__metaclass__ = abc.ABCMeta

	def __init__(self):
		pass

	@abc.abstractmethod
	def get_result(self, loan_id):
		pass

	def parse(self, loan_id):
		f = open('media/credit_risk/dataset/test_id_dataset.csv', 'r')
		line = f.readline()
		sp = line.split(',')
		count = 0
		while(line != ""):
			if(sp[0] == loan_id):
				break
			sp = line.split(',')
			line = f.readline()
		del sp[0]
		del sp[0]
		return sp

	@abc.abstractmethod
	def load_model(self, model):
		pass

	def preprocess(self, arr):
		reloaded = self.model
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

class StatisticalStrategy(DataBasedStrategyAbstract):
	def load_model(self):
		f = open('media/credit_risk/dataset/statistical.save', 'rb')
		reloaded = cPickle.load(f)
		f.close()
		self.model = reloaded

	def get_result(self, loan_id):
		self.load_model()
		super(StatisticalStrategy, self).preprocess(np.array(self.parse(loan_id)))
		result = self.model.trained_model.predict_proba(self.df)
		return (str(result[0][0])+ ","+str(result[0][1]))

class MLStrategy(DataBasedStrategyAbstract):
	def load_model(self):
		f = open('media/credit_risk/dataset/ml.save', 'rb')
		reloaded = cPickle.load(f)
		f.close()
		self.model = reloaded

	def get_result(self, loan_id):
		self.load_model()
		super(MLStrategy, self).preprocess(np.array(self.parse(loan_id)))
		result = self.model.trained_model.predict_proba(self.df)
		return (str(result[0][0])+ ","+str(result[0][1]))

rule_strategy = RuleBasedStrategy()
stat_strategy = StatisticalStrategy()
ml_strategy = MLStrategy()

class Classifier(object):
	def __init__(self, strategy):
		self._strategy = strategy

	def get_result(self, loan_id):
		return self._strategy.get_result(loan_id)

class RuleClassifier(Classifier):
	def __init__(self):
		super(RuleClassifier, self).__init__(rule_strategy)

class StatisticalClassifier(Classifier):
	def __init__(self):
		super(StatisticalClassifier, self).__init__(stat_strategy)

class MLClassifier(Classifier):
	def __init__(self):
		super(MLClassifier, self).__init__(ml_strategy)
