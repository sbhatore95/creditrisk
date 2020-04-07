from django.shortcuts import render, redirect
from django.views.decorators.http import require_POST
from django.db import models
from django.views.generic import FormView
from django.urls import reverse
from urllib.parse import urlencode
from .forms import MyForm, FileUploadForm
from .project import *
import sys
from .predict import *
from .models import SavedState

def index(request):
	form = MyForm()
	ins = SavedState.objects.all().first()
	status_bg = False
	if(ins is not None):
		status_bg = ins.stat == "true"
	context = {'form':form, 'status_bg':status_bg}
	return render(request, 'loan_officer/index.html', context)

def result(request):
	loan_id = request.GET.get('loan_id')
	rule = request.GET.get('rule_based')
	stat = request.GET.get('statistical_based')
	ml = request.GET.get('ML_based')
	if(SavedState.objects.all().first() == None):
		m = SavedState(stat="false", ml="false")
		m.save()
	if((stat or ml) and SavedState.objects.all().first().stat == "false"):
		return render(request, 'loan_officer/result.html', {'not_ready':True})
	classifier = Classifier(rule, stat, ml)
	ans = classifier.doClassification(loan_id)
	stat_approve = None
	stat_napprove = None
	ml_approve = None
	ml_napprove = None
	if(ans[1]):
		stat_approve = ans[1].split(',')[0]
		stat_napprove = ans[1].split(',')[1]
	if(ans[2]):
		ml_approve = ans[2].split(',')[0]
		ml_napprove = ans[2].split(',')[1]
	context = {'rule': ans[0], 'stat_approve': stat_approve, 'stat_napprove': stat_napprove, 
	'ml_approve': ml_approve, 'ml_napprove': ml_napprove, 'statandml':ans[3]}
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
		form.process_data(request.FILES['file'])
		base_url = reverse('loan_officer:uploadCSV')
		query_string =  urlencode({'add': 'ok'})
		url = '{}?{}'.format(base_url, query_string)
	return redirect(url)