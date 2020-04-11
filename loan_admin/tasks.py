from loan_officer.models import SavedState
from loan_officer.project import *
from .models import UploadFile
from .data_driver import *

def bg_task():
	if(SavedState.objects.all().first() == None or SavedState.objects.all().first().ml == "true"):
		print("Inside bg_task()")
		ins = UploadFile.objects.all().first()
		cols = ins.columns.split(',')
		del cols[0]
		target = ins.target
		noml = ins.nominal_features.split(',')
		print("---Learning started---")
		DataDriver.process("media/credit_risk/dataset/id_dataset.csv", 
			"media/credit_risk/dataset/dataset.csv")
		DataDriver.learn_and_save(cols, target, noml, "media/credit_risk/dataset/dataset.csv")
		print("---Learned and Saved---")
		if(SavedState.objects.all().first() == None):
			m = SavedState(stat="true", ml="false", statandml="true")
		else:
			m = SavedState.objects.all().first()
			m.stat = "true"
		m.save()