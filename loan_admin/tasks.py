from loan_officer.models import SavedState
from loan_officer.project import *
from background_task import background

@background(schedule=1)
def bg_task(cols, noml, dataset):
	print("Inside bg_task()")
	learn_and_save("statistical", cols, noml, dataset)
	print("statistical done")
	learn_and_save("ml", cols, noml, dataset)
	print("ml done")
	learn_and_save("statandml", cols, noml, dataset)
	print("statandml done")
	if(SavedState.objects.all().first() == None):
		m = SavedState(stat="true", ml="true", statandml="true")
	else:
		m = SavedState.objects.all().first()
		m.stat = "true"
	m.save()