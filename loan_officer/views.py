from django.shortcuts import render, redirect
from django.views.decorators.http import require_POST
from django.db import models
from django.views.generic import FormView
from django.urls import reverse
from urllib.parse import urlencode
from .forms import MyForm, FileUploadForm
from .project import *
import sys
from .models import SavedState
import pickle
import codecs
from six.moves import cPickle
from loan_admin.models import UploadFile, Criteria, CriteriaHelper, Configuration

def index(request):
	form = MyForm()
	status_bg = SavedState.objects.all().first().stat == "true"
	context = {'form':form, 'status_bg':status_bg}
	return render(request, 'loan_officer/index.html', context)
	# x = predict_score(columns, 'loan_status', nominal, "dataset.csv")
	# print(x.Preprocess())

def result(request):
	loan_id = request.GET.get('loan_id')
	rule = request.GET.get('rule_based')
	stat = request.GET.get('statistical_based')
	ml = request.GET.get('ML_based')
	res = ""
	columns = UploadFile.objects.all().first().columns.split(',')
	del columns[0]
	nominal = UploadFile.objects.all().first().nominal_features.split(',')
	if(SavedState.objects.all().first() == None):
		m = SavedState(stat="false", ml="false")
		m.save()
	if(rule and stat and ml):
		pass
	elif(rule and stat):
		pass
	elif(rule and ml):
		pass
	elif(stat and ml):
		if(SavedState.objects.all().first().statandml == "false"):
			learn_and_save("statandml", columns, nominal, "dataset.csv")
			m = SavedState.objects.all().first()
			m.statandml = "true"
			m.save()
		f = open('media/credit_risk/dataset/test_id_dataset.csv', 'r')
		line = f.readline()
		sp = line.split(',')
		count = 0
		while(line != ""):
			if(sp[0] == loan_id):
				break
			sp = line.split(',')
			line = f.readline()
		# if(sp[0] != int(loan_id)):
		del sp[0]
		del sp[0]
		res = load_and_predict("statandml", sp)
	elif(rule):
		f = open('media/credit_risk/dataset/test_id_dataset.csv', 'r')
		line = f.readline()
		array = line.split(',')
		array[-1] = array[-1].replace('\n', '')
		print(array)
		Dict = {}
		for i in range(0, len(array)):
			Dict[array[i]] = i
		print(Dict)
		line = f.readline()
		while line != "":
			sp = line.split(',')
			if(sp[0] == loan_id):
				break
			line = f.readline()
		sp[-1] = sp[-1].replace('\n', '')
		print(sp)
		cri = Criteria.objects.all()
		conf = Configuration.objects.all()
		ans = 0
		for configuration in conf:
			for criteria in cri:
				helper = CriteriaHelper.objects.all().filter(criteria=criteria)
				truth = (criteria.feature == configuration.feature and criteria.product == configuration.product
					and criteria.category == configuration.category)
				print(truth)
				if(truth):
					print(criteria.data_source == '3')
					if(criteria.data_source == '3'): # for SQL
						feature = criteria.api.split(' ')[-1]
						print(feature)
					feature_score = 0
					for crhelper in helper:
						# For categorical
						tup = crhelper.entry.split(' ')
						if(tup[0] == "is"):
							print(sp[Dict[feature]])
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
	elif(stat):
		if(SavedState.objects.all().first().stat == "false"):
			learn_and_save("statistical", columns, nominal, "dataset.csv")
			m = SavedState.objects.all().first()
			m.stat = "true"
			m.save()
		f = open('media/credit_risk/dataset/test_id_dataset.csv', 'r')
		line = f.readline()
		sp = line.split(',')
		count = 0
		while(line != ""):
			if(sp[0] == loan_id):
				break
			sp = line.split(',')
			line = f.readline()
		# if(sp[0] != int(loan_id)):
		del sp[0]
		del sp[0]
		res = load_and_predict("statistical", sp)
	else:
		if(SavedState.objects.all().first().ml == "false"):
			learn_and_save("ml", columns, nominal, "dataset.csv")
			m = SavedState.objects.all().first()
			m.ml = "true"
			m.save()
		f = open('media/credit_risk/dataset/test_id_dataset.csv', 'r')
		line = f.readline()
		sp = line.split(',')
		count = 0
		while(line != ""):
			if(sp[0] == loan_id):
				break
			sp = line.split(',')
			line = f.readline()
		# if(sp[0] != int(loan_id)):
		f.close()
		del sp[0]
		del sp[0]
		res = load_and_predict("ml", sp)
	if(rule):
		context = {'status':True,'ans':ans}
		return render(request, 'loan_officer/result.html', context)
	else:
		out = res.split(',')
		context = {'status':False, 'approve': out[0], 'not_approve': out[1]}
		return render(request, 'loan_officer/result.html', context)

def uploadCSV(request):
	add = request.GET.get('add')
	form = FileUploadForm()
	context = {'form':form, 'add':add}
	return render(request, 'loan_officer/uploadCSV.html', context)

@require_POST
def addApplicant(request):
	form = FileUploadForm(request.POST, request.FILES)
	url = reverse('loan_officer:uploadCSV')
	if form.is_valid():
		# form.process_data(request.POST, request.FILES['file'])
		form.save()
		base_url = reverse('loan_officer:uploadCSV')
		query_string =  urlencode({'add': 'ok'})
		url = '{}?{}'.format(base_url, query_string)
	return redirect(url)