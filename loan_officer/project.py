#!/usr/bin/env python
# coding: utf-8

import pandas as pd # type: ignore
import numpy as np
from sklearn import preprocessing # type: ignore
from sklearn.preprocessing import OneHotEncoder # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from imblearn.over_sampling import SMOTE # type: ignore
from sklearn.neural_network import MLPClassifier # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.metrics import classification_report,confusion_matrix # type: ignore
from sklearn.neighbors import KNeighborsClassifier # type: ignore
from sklearn.model_selection import GridSearchCV # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
#import xgboost as xgb
from sklearn import svm # type: ignore
from sklearn.ensemble import IsolationForest # type: ignore
import warnings
from sklearn.metrics import plot_confusion_matrix # type: ignore
from sklearn.decomposition import PCA # type: ignore
import pickle 
import codecs
from six.moves import cPickle # type: ignore

# columns = ['loan_status','funded_amnt','term','int_rate','installment','sub_grade','emp_length',
# 'home_ownership','annual_inc','verification_status','pymnt_plan','purpose',
# 'dti','delinq_2yrs','inq_last_6mths','open_acc','pub_rec','revol_bal','revol_util','total_acc',
# 'initial_list_status','collections_12_mths_ex_med','acc_now_delinq','tot_coll_amt','acc_open_past_24mths',
# 'avg_cur_bal','bc_open_to_buy','chargeoff_within_12_mths','delinq_amnt','mo_sin_old_il_acct',
# 'mo_sin_old_rev_tl_op','mo_sin_rcnt_rev_tl_op','mo_sin_rcnt_tl','mort_acc','mths_since_recent_bc',
# 'mths_since_recent_inq','num_accts_ever_120_pd',
# 'num_actv_bc_tl','num_bc_tl','num_il_tl','num_tl_90g_dpd_24m','num_tl_op_past_12m','pct_tl_nvr_dlq',
# 'percent_bc_gt_75','pub_rec_bankruptcies','tax_liens','total_il_high_credit_limit']
# #qualitative = ['status', 'credit_history', 'purpose','savings', 'employment','status_gender', 
# #            'guarantors','property', 'age','other_installment', 'housing','job', 'phone', 'foreign_worker', 'approve']
# #numerical = ['duration','credit_amount','installment_rate','residence','existing_credits', 'liable']
# #nominal = ['credit_history', 'purpose', 'status_gender', 'guarantors', 'property', 'other_installment', 'housing', 'job']
# numerical = ['loan_status','funded_amnt','annual_inc','dti','delinq_2yrs','inq_last_6mths',
# 'pub_rec','revol_bal','revol_util','total_acc','collections_12_mths_ex_med','acc_now_delinq','tot_coll_amt',
# 'acc_open_past_24mths',
# 'avg_cur_bal','bc_open_to_buy','chargeoff_within_12_mths','delinq_amnt','mo_sin_old_il_acct',
# 'mo_sin_old_rev_tl_op','mo_sin_rcnt_rev_tl_op','mo_sin_rcnt_tl','mort_acc','mths_since_recent_bc',
# 'mths_since_recent_inq','mths_since_recent_revol_delinq','num_accts_ever_120_pd',
# 'num_actv_bc_tl','num_bc_tl','num_il_tl','num_tl_90g_dpd_24m','num_tl_op_past_12m','pct_tl_nvr_dlq',
# 'percent_bc_gt_75','pub_rec_bankruptcies','tax_liens','total_il_high_credit_limit']
# nominal = ['term','sub_grade','emp_length','home_ownership','verification_status',
# 'pymnt_plan','purpose','initial_list_status']
# column_arr = []
# curr_column = ['loan_status']
# nominal_arr = []
# nominal_curr = []
# for i in range(1, len(columns)):
#     curr_column.append(columns[i])
#     column_arr.append(list(curr_column))
#     if columns[i] in nominal:
#         nominal_curr.append(columns[i])
#     nominal_arr.append(list(nominal_curr))
# print(column_arr[1])


# In[131]:


class predict_score:
    top_model = ""
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
        self.Best_Model = ""
        self.Best_Acc = 0
        
    def Info(self):
        print(self.dataset.info())
    
    def Unique(self):
        print(self.dataset.nunique())        
    
    def Label_Encoding(self):
        Cat_dataset = self.dataset.select_dtypes(include=[object])
        for col in Cat_dataset:
            # print(Cat_dataset[col])
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
#         print(conf_mat)
        #print("Confusion matrix : ",conf_mat)
        weighted_accuracy = (0.7 *(conf_mat[1][1])/sum(conf_mat[1])) + (0.3*(conf_mat[0][0]/sum(conf_mat[0]))) 
        accuracy = (conf_mat[1][1] + conf_mat[0][0])/(sum(conf_mat[1]) + sum(conf_mat[0]))
        return weighted_accuracy, accuracy

    #Doesn't depend on parameteres
    def KNN(self):
#         print("--------- KNN ---------")
        model = KNeighborsClassifier()
        params = {'n_neighbors':[i for i in range(1,50,2)]}
        self.KNN_model = GridSearchCV(model, param_grid=params,cv=10,scoring='f1')
        self.KNN_model.fit(self.X_train,self.Y_train)
        # print("Best Hyper Parameters:",self.KNN_model.best_params_)
        Y_pred = self.KNN_model.predict(self.X_test)
        wt_ac, ac = self.weighted_accuracy(self.Y_test, Y_pred)
        if(self.Best_Acc < ac):
            self.Best_Acc = ac
            self.Best_Model = "KNN_model"
        return ("For KNN -> Weighted accuracy : "+str(wt_ac)+", Normal accuracy : "+str(ac))
        
    
    def SVM(self):
#         print("--------- SVM ------------")
        # C = [0.01, 0.1, 1, 10, 100]
        C = [0.1, 1, 10]
        # kernels = ['linear', 'rbf', 'poly']
        kernels = ['rbf']
        param_grid = {'kernel':kernels, 'C':C}
        print("1")
        self.svm_model = GridSearchCV(svm.SVC(probability = True), param_grid, cv=10, scoring='f1')
        print("2")
        self.svm_model.fit(self.X_train, self.Y_train)
        print(self.svm_model.best_params_)
        Y_pred = self.svm_model.predict(self.X_test)
        print("3")
        wt_ac, ac = self.weighted_accuracy(self.Y_test, Y_pred)
        print("Weighted accuracy : "+str(wt_ac)+", Normal accuracy : "+str(ac))

        if(self.Best_Acc < ac):
            self.Best_Acc = ac
            self.Best_Model = "svm_model"

    def Logistic(self):
#         print("-------- Logistic Regression ---------")
        warnings.filterwarnings("ignore")
        param_grid = {'C': [0.01, 0.1, 1,10,100] ,'penalty' : ['l1', 'l2']}
        self.Logistic_model = GridSearchCV(LogisticRegression(), param_grid, cv=10, scoring='f1')
        self.Logistic_model.fit(self.X_train, self.Y_train)
        best_params = self.Logistic_model.best_params_
        Y_pred = self.Logistic_model.predict(self.X_test)
        wt_ac, ac = self.weighted_accuracy(self.Y_test, Y_pred)
        best_params = " " + "Weighted accuracy : "+str(wt_ac)+", Normal accuracy : "+str(ac)
        if(self.Best_Acc < ac):
            self.Best_Acc = ac
            self.Best_Model = "Logistic_model"
            # predict_score.top_model = "Logistic_model"
        return best_params
    
    def Neural_Network(self):
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
        
        self.NN_model = GridSearchCV(Mlp, parameter_space, n_jobs = -1, cv = 10)
        self.NN_model.fit(self.X_train, self.Y_train)
        print('Best parameters found:\n', self.NN_model.best_params_)
        Y_pred = self.NN_model.predict(self.X_test)
        wt_ac, ac = self.weighted_accuracy(self.Y_test, Y_pred)
        print("Weighted accuracy : "+str(wt_ac)+", Normal accuracy : "+str(ac))
        if(self.Best_Acc < ac):
            self.Best_Acc = ac
            self.Best_Model = "NN_model"
   
    def Random_forest(self):
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
        self.Random_forest_model.fit(self.X_train, self.Y_train)
        print('Best parameters found:\n', self.Random_forest_model.best_params_)
        Y_pred = self.Random_forest_model.predict(self.X_test)
        wt_ac, ac = self.weighted_accuracy(self.Y_test, Y_pred)
        print("Weighted accuracy : "+str(wt_ac)+", Normal accuracy : "+str(ac))
        if(self.Best_Acc < ac):
            self.Best_Acc = ac
            self.Best_Model = "Random_forest"
        
    def Prepare_Data(self, User_Input):
        self.df = pd.DataFrame(User_Input.reshape(-1, len(User_Input)), columns = self.Test_columns)
                
        for col in self.le_name_mapping:
            self.df[col] = self.le_name_mapping[col][self.df[col][0]]
       
        if(len(self.Nominal_Converted_features) > 0):
            z = np.zeros(len(self.Nominal_Converted_features), dtype = int)
            nominal_df = pd.DataFrame(z.reshape(-1, len(z)), columns = self.Nominal_Converted_features)

            for col in self.Nominal_Features:
                nominal_df[col+"_"+str(self.df[col][0])] = 1
                self.df = self.df.drop(col, 1)
            self.df = pd.concat([self.df, nominal_df], axis = 1)
        
        #Standardization of data frame
        
        df_std = pd.DataFrame(self.std_scale.transform(self.df[self.Num_Features]), columns = self.Num_Features)
        self.df = self.df.drop(self.Num_Features, axis = 1)
        self.df = pd.concat([self.df,df_std], axis = 1)
        print("-->" + self.Best_Model)
        if(self.Best_Model == "KNN_model"):
            result = self.KNN_model.predict_proba(self.df)
        
        elif(self.Best_Model == "svm_model"):
            result = self.svm_model.predict_proba(self.df)
        
        elif(self.Best_Model == "Logistic_model"):
            result = self.Logistic_model.predict_proba(self.df)
        
        elif(self.Best_Model == "NN_model"):
            result = self.NN_model.predict_proba(self.df)
        
        elif(self.Best_Model == "Random_forest"):
            result = self.Random_forest_model.predict_proba(self.df)
            return (str(result[0][0])+ ","+str(result[0][1]))
        
    def Preprocess(self, model):
            self.Label_Encoding()
            self.One_Hot_Encoding()
            self.Outlier()
            self.Split_Data(0.2)
            self.Standardize()
            if(model == "statistical"):
                print(self.Logistic())
            elif(model == "ml"):
                print(self.KNN())
            else:
                print(self.Logistic())
                print(self.KNN())

def learn_and_save(model, columns, nominal, dataset):
    x = predict_score(columns, 'loan_status', nominal, dataset)
    if(model == "statistical"):
        x.Preprocess("statistical")
        f = open('statistical.save', 'wb')
        pickled = cPickle.dump(x, f)
        f.close()
    elif(model == "ml"):
        x.Preprocess("ml")
        # print("-<>" + x.Best_Model)
        f = open('ml.save', 'wb')
        pickled = cPickle.dump(x, f)
        f.close()
    elif(model == "statandml"):
        x.Preprocess("statandml")
        f = open('statandml.save', 'wb')
        pickled = cPickle.dump(x, f)
        f.close()

def load_and_predict(model, arr):
    reloaded = ""
    if(model == "statistical"):
        f = open('statistical.save', 'rb')
        reloaded = cPickle.load(f)
        f.close()
    elif(model == "ml"):
        f = open('ml.save', 'rb')
        reloaded = cPickle.load(f)
        f.close()
    elif(model == "statandml"):
        f = open('statandml.save', 'rb')
        reloaded = cPickle.load(f)
        f.close()
    m = np.array(arr)
    return reloaded.Prepare_Data(m)

