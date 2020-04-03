from loan_officer.models import SavedState
from loan_officer.project import *
from .models import UploadFile

def bg_task():
	if(SavedState.objects.all().first() == None or SavedState.objects.all().first().ml == "true"):
		print("Inside bg_task()")
		cols = UploadFile.objects.all().first().columns.split(',')
		del cols[0]
		noml = UploadFile.objects.all().first().nominal_features.split(',')
		learn_and_save("statistical", cols, noml, "dataset.csv")
		print("statistical done")
		learn_and_save("ml", cols, noml, "dataset.csv")
		print("ml done")
		learn_and_save("statandml", cols, noml, "dataset.csv")
		print("statandml done")
		if(SavedState.objects.all().first() == None):
			m = SavedState(stat="true", ml="false", statandml="true")
		else:
			m = SavedState.objects.all().first()
			m.stat = "true"
		m.save()
	# print("heyheyheyhey")
	# f = open('a.csv', 'w')
	# f.write("hey!!")