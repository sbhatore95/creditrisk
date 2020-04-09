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

class Trainer(metaclass=ABCMeta):
	"""Base trainer to be used for all models."""

	def __init__(self, Column_Names, Output_Feature, Nominal_Features, Dataset_Location):
		self.Column_Names = Column_Names
		self.Test_columns = Column_Names[1:]
		self.dataset = pd.read_csv(Dataset_Location, sep=",", names=self.Column_Names, usecols=self.Column_Names)
		self.Nominal_Features = Nominal_Features
		self.Output_Feature = Output_Feature
		self.Num_Features = list(self.dataset.select_dtypes(include = [int, float]))
		self.Num_Features.remove(self.Output_Feature)
		self.Cat_Features = list(self.dataset.select_dtypes(include = [object]))
		self.Le = preprocessing.LabelEncoder()
		self.Approve_dataset = self.dataset.loc[self.dataset[Output_Feature] == 0]
		self.Not_Approve_dataset = self.dataset.loc[self.dataset[Output_Feature] != 0]
		self.le_name_mapping = {}
		self.Nominal_Converted_features = []
		self.X = pd.DataFrame()
		self.Y = pd.DataFrame()
		self.X_train = pd.DataFrame()
		self.X_test = pd.DataFrame()
		self.Y_train = pd.DataFrame()
		self.Y_test = pd.DataFrame()
		self.acc = 0
		self.trained_model = ""

	def Label_Encoding(self):
		Cat_dataset = self.dataset.select_dtypes(include=[object])
		for col in Cat_dataset:
			self.dataset[col] = self.Le.fit_transform(Cat_dataset[col])
			self.le_name_mapping[col]  = dict(zip(self.Le.classes_, self.Le.transform(self.Le.classes_)))
		
	def One_Hot_Encoding(self):
		for col in self.Nominal_Features:
			temp = pd.get_dummies(data = self.dataset[col], prefix = col)
			self.Nominal_Converted_features.extend(list(temp.columns))
			self.dataset = pd.concat([self.dataset, temp], axis = 1)
			self.dataset = self.dataset.drop(col, axis = 1)
	   
   
	def Outlier(self):
		clf=IsolationForest(n_estimators=100, max_samples='auto',  contamination='auto',                         max_features=1.0, bootstrap=False, n_jobs=-1, random_state=42, behaviour = "new", verbose=0)
		clf.fit(self.dataset)
		pred = clf.predict(self.dataset)
		self.dataset['anomaly'] = pred
		self.dataset.drop(self.dataset.loc[self.dataset['anomaly']==-1].index, inplace = True)
		self.dataset = self.dataset.drop('anomaly', axis = 1)
		
	def Split_Data(self, test_split): 
		self.X = self.dataset.drop(self.Output_Feature, 1)
		self.Y = self.dataset[self.Output_Feature]
		X_train_temp, self.X_test, Y_train_temp, self.Y_test = train_test_split(self.X, self.Y, test_size = test_split, random_state = 42)
		self.X_test.reset_index(inplace = True)
		self.X_test.drop(self.X_test.columns[0], axis = 1, inplace = True)
		#Resolving output imbalance
		sm = SMOTE(random_state = 42)
		X_train_arr, self.Y_train = sm.fit_sample(X_train_temp, Y_train_temp)
		self.X_train = pd.DataFrame(X_train_arr, columns = self.X.columns)


	def Standardize(self):
		self.std_scale = preprocessing.StandardScaler().fit(self.X_train[self.Num_Features])
		train_df_std = pd.DataFrame(self.std_scale.transform(self.X_train[self.Num_Features]), columns = self.Num_Features)
		self.X_train = self.X_train.drop(self.Num_Features, axis = 1)
		self.X_train = pd.concat([self.X_train, train_df_std], axis = 1)
		test_df_std = pd.DataFrame(self.std_scale.transform(self.X_test[self.Num_Features]), columns = self.Num_Features)
		self.X_test = self.X_test.drop(self.Num_Features, axis = 1)
		self.X_test = pd.concat([self.X_test, test_df_std], axis = 1)

	
	#Weighted accuracy calculation from confusion matrix
	def weighted_accuracy(self, Y_test, prediction):
		conf_mat = confusion_matrix(Y_test,prediction)
		weighted_accuracy = (0.7 *(conf_mat[1][1])/sum(conf_mat[1])) + (0.3*(conf_mat[0][0]/sum(conf_mat[0]))) 
		accuracy = (conf_mat[1][1] + conf_mat[0][0])/(sum(conf_mat[1]) + sum(conf_mat[0]))
		return weighted_accuracy, accuracy

	def preprocess(self):
		self.Label_Encoding()
		self.One_Hot_Encoding()
		self.Outlier()
		self.Split_Data(0.2)
		self.Standardize()

	@abstractmethod
	def set_model(self):
		"""Define model here."""

	def fit_model(self):
		self.trained_model.fit(self.X_train,self.Y_train)

	def generate_metrics(self):
		Y_pred = self.trained_model.predict(self.X_test)
		wt_ac, ac = super(KNN, self).weighted_accuracy(self.Y_test, Y_pred)
		self.acc = ac
		return ac

	def save_model(self, model_name):
		"""This method saves the model in our required format."""

class KNN(Trainer):
	def __init__(self, Column_Names, Output_Feature, Nominal_Features, Dataset_Location):
		super(KNN, self).__init__(Column_Names, Output_Feature, Nominal_Features, 
			Dataset_Location)
	
	def set_model(self):
		model = KNeighborsClassifier()
		params = {'n_neighbors':[i for i in range(1,50,2)]}
		self.trained_model = GridSearchCV(model, param_grid=params,cv=10,scoring='f1')

class Logistic(Trainer):
	def __init__(self, Column_Names, Output_Feature, Nominal_Features, Dataset_Location):
		super(Logistic, self).__init__(Column_Names, Output_Feature, Nominal_Features, 
			Dataset_Location)

	def set_model(self):
		warnings.filterwarnings("ignore")
		param_grid = {'C': [0.01, 0.1, 1,10,100] ,'penalty' : ['l1', 'l2']}
		self.trained_model = GridSearchCV(LogisticRegression(), param_grid, cv=10, scoring='f1')

class SVM(Trainer):
	def __init__(self, Column_Names, Output_Feature, Nominal_Features, Dataset_Location):
		super(SVM, self).__init__(Column_Names, Output_Feature, Nominal_Features, 
			Dataset_Location)

	def set_model(self):
		C = [0.1, 1, 10]
		# kernels = ['linear', 'rbf', 'poly']
		kernels = ['rbf']
		param_grid = {'kernel':kernels, 'C':C}
		print("1")
		self.trained_model = GridSearchCV(svm.SVC(probability = True), param_grid, 
			cv=10, scoring='f1')

class NN_Network(Trainer):
	def __init__(self, Column_Names, Output_Feature, Nominal_Features, Dataset_Location):
		super(NN_Network, self).__init__(Column_Names, Output_Feature, Nominal_Features, 
			Dataset_Location)

	def set_model(self):
		print("------- Neural Network ---------")
		warnings.filterwarnings("ignore")
		Input = len(self.dataset.columns)
		Mlp = MLPClassifier()
		parameter_space = {
		#'hidden_layer_sizes': [(Input,100,2), (Input,100,25,2),(Input,100,50,25,2)],
		# 'hidden_layer_sizes' : np.arange(5, 12),
		'hidden_layer_sizes' : np.arange(6, 10),
		# 'solver': ['sgd', 'adam', 'lbfgs'],
		'solver': ['adam'],
		# 'alpha':10.0 ** -np.arange(1,7),
		'alpha':10.0 ** -np.arange(1,4),
		# 'batch_size' : [100,200,300,400,500],
		'batch_size' : [200],
		# 'learning_rate': ['constant','adaptive'],}
		'learning_rate': ['constant'],}
		self.trained_model = GridSearchCV(Mlp, parameter_space, n_jobs = -1, cv = 10)

class RandomForest(Trainer):
	def __init__(self, Column_Names, Output_Feature, Nominal_Features, Dataset_Location):
		super(RandomForest, self).__init__(Column_Names, Output_Feature, Nominal_Features, 
			Dataset_Location)

	def set_model(self):
		print("------- Random Forest ---------")
        rfc = RandomForestClassifier()
        param_grid = { 
            # 'n_estimators': [200, 500],
            'n_estimators': [500],
            # 'max_features': ['auto', 'sqrt', 'log2'],
            'max_features': ['auto'],
            # 'max_depth' : [4,5,6,7,8],
            'max_depth' : [8],
            # 'criterion' :['gini', 'entropy']
            'criterion' :['gini']
        }
        self.Random_forest_model = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 10)
