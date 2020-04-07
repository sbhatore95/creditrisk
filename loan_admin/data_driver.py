from .trainer import *
from .data_processor import *
from loan_officer.models import SavedState

class DataDriver:
	def __init__(self):
		pass

	def learn_and_save(columns, target, nominal, dataset):
		tr_knn = KNN(columns, target, nominal, dataset)
		tr_lr = Logistic(columns, target, nominal, dataset)
		models_ml = [tr_knn]
		models_statistical = [tr_lr]	    
		acc_ml = []
		acc_statistical = []
		print("---ML started---")
		for model in models_ml:
			model.preprocess()
			model.set_model()
			model.fit_model()
			acc_ml.append(model.generate_metrics())
		print("---ML ended---")
		print("---Statistical started---")
		for model in models_statistical:
			model.preprocess()
			model.set_model()
			model.fit_model()
			acc_statistical.append(model.generate_metrics())
			print("---Statistical ended---")
		acc_highest_stat = 0
		for acc in acc_statistical:
			if(float(acc) > acc_highest_stat):
				acc_highest_stat = float(acc)
		to_save = models_statistical[acc_statistical.index(acc_highest_stat)]
		f = open('media/credit_risk/dataset/statistical.save', 'wb')
		pickled = cPickle.dump(to_save, f)
		f.close()
		acc_highest_ml = 0
		for acc in acc_ml:
			if(float(acc) > acc_highest_ml):
				acc_highest_ml = float(acc)
		to_save = models_ml[acc_ml.index(acc_highest_ml)]
		f = open('media/credit_risk/dataset/ml.save', 'wb')
		pickled = cPickle.dump(to_save, f)
		f.close()
		
		# Coupling spotted
		ins = SavedState.objects.all().first()
		if(acc_highest_stat > acc_highest_ml):
			ins.statandml = "stat"
		else:
			ins.statandml = "ml"
		ins.save()
