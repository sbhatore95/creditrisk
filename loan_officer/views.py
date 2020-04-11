from django.shortcuts import render, redirect
from django.views.decorators.http import require_POST
from django.db import models
from django.views.generic import FormView
from django.urls import reverse
from urllib.parse import urlencode
from .forms import MyForm, FileUploadForm
from .predict import *
from .models import SavedState
from login.models import Sessions
# from notifications import notify

def get_params(request, keys=[]):
	ans = {}
	for key in keys:
		ans[key] = request.GET.get(key)
	ans['session'] = Sessions.objects.all().first().user
	ans['savedstate'] = SavedState.objects.all().first()
	return ans

def index(request):
	form = MyForm()
	params = get_params(request)
	status_bg = False
	if(params['savedstate'] is not None):
		status_bg = (params['savedstate'].stat == "true")
	context = {'form':form, 'status_bg':status_bg, 'session': params['session']}
	return render(request, 'loan_officer/index.html', context)

def result(request):
	params = get_params(request, ['loan_id', 'rule_based', 'statistical_based', 'ML_based'])
	if(params['savedstate'] == None):
		m = SavedState(stat="false", ml="false")
		m.save()
	if((params['statistical_based'] or params['ML_based']) and params['savedstate'].stat == "false"):
		return render(request, 'loan_officer/result.html', {'not_ready':True})
	ans = get_results(params['rule_based'], params['statistical_based'], params['ML_based'], params['loan_id'])
	arr = []
	arr.append(ans[0])
	arr += result_helper(ans)
	arr.append(ans[3])
	context = {'result': arr, 'session':params['session']}
	return render(request, 'loan_officer/result.html', context)

def result_helper(ans):
	arr = [None, None, None, None]
	if(ans[1]):
		arr[0] = ans[1].split(',')[0]
		arr[1] = ans[1].split(',')[1]
	if(ans[2]):
		arr[2] = ans[2].split(',')[0]
		arr[3] = ans[2].split(',')[1]
	return arr

def get_results(rule, stat, ml, loan_id):
	ans = [None, None, None, None]
	if(rule):			
		classifier = RuleClassifier()
		ans[0] = classifier.get_result(loan_id)
	if(stat):
		classifier = StatisticalClassifier()
		ans[1] = classifier.get_result(loan_id)
	if(ml):
		classifier = MLClassifier()
		ans[2] = classifier.get_result(loan_id)
	if(stat and ml):
		ans[3] = (SavedState.objects.all().first().statandml == 'stat')
	return ans

def uploadCSV(request):
	params = get_params(request, ['add'])
	form = FileUploadForm()
	if(params['add'] == 'ok'):
		messages.info(request, 'Record created successfully')
	context = {'form':form, 'session': params['session']}
	return render(request, 'loan_officer/uploadCSV.html', context)

@require_POST
def addApplicant(request):
	form = FileUploadForm(request.POST, request.FILES)
	url = reverse('loan_officer:uploadCSV')
	if form.is_valid():
		form.process_data(request.FILES['file'])
		base_url = reverse('loan_officer:uploadCSV')
		query_string =  urlencode({'add': 'ok'})
		url = '{}?{}'.format(base_url, query_string)
	return redirect(url)


