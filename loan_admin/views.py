from django.shortcuts import render, redirect
from django.views.decorators.http import require_POST

from .models import Feature
from .models import Configuration
from .models import Criteria, CriteriaHelper, SetScale
from .forms import FeatureForm
from .forms import ConfigurationForm

from django.urls import reverse
from urllib.parse import urlencode
from .forms import UploadFileForm, CriteriaForm, SetScaleForm
import sys
from django.contrib import messages
from django.http import JsonResponse
from login.models import Sessions
# Create your views here.
def get_params(request, key):
	add = request.GET.get(key)
	session = Sessions.objects.all().first()
	return {'add':add, 'session':session.user}

def index(request):
	params = get_params(request, 'add')
	if(params['add'] == 'ok1'):
		messages.info(request, 'Record created successfully')
	form = FeatureForm()
	context = {'form':form, 'session':params['session']}
	return render(request, 'loan_admin/index.html', context)

def configuration(request):
	params = get_params(request, 'add1')
	form = ConfigurationForm()
	if(params['add'] == 'ok2'):
		messages.info(request, 'Record created successfully')
	context = {'form':form, 'session':params['session']}
	return render(request, 'loan_admin/configuration.html', context)

def get_feature_values(request):
	name = request.GET.get('name')
	ins = Feature.objects.filter(name=name).first()
	data = {
		'value': ins.value,
		'data_type': ins.data_type,
		'category': ins.category,
		'status': ins.status
	}
	return JsonResponse(data)

@require_POST
def addFeature(request):
	form = FeatureForm(request.POST)
	url = reverse('loan_admin:index') 
	if form.is_valid():
		feature = form.save()
		base_url = reverse('loan_admin:index')
		query_string =  urlencode({'add': 'ok1'})
		url = '{}?{}'.format(base_url, query_string)
	return redirect(url)

@require_POST
def addConfiguration(request):
	form = ConfigurationForm(request.POST)
	url = reverse('loan_admin:configuration') 
	if form.is_valid():
		feature = form.save()
		base_url = reverse('loan_admin:configuration')
		query_string =  urlencode({'add1': 'ok2'})
		url = '{}?{}'.format(base_url, query_string)
	return redirect(url)

def get_configuration_values(request):
	feature = request.GET.get('feature')
	ins = Configuration.objects.filter(feature=feature).first()
	data = {
		'product': ins.product,
		'weightage': ins.weightage,
		'category': ins.category
	}
	return JsonResponse(data)

def criteria(request):
	params = get_params(request, 'add2')
	form = CriteriaForm()
	if(params['add'] == 'ok3'):
		messages.info(request, 'Record created successfully')
	context = {'form':form, 'session':params['session']}
	return render(request, 'loan_admin/criteria.html', context)

@require_POST
def addCriteria(request):
	print("----------")
	print(request.POST.get('entry_2'))
	print("----------")
	form = CriteriaForm(request.POST)
	url = reverse('loan_admin:criteria') 
	print("++++")
	if form.is_valid():
		form.clean()
		print("?????")
		form.save()
		print("*****")
		base_url = reverse('loan_admin:criteria')
		query_string =  urlencode({'add2': 'ok3'})
		url = '{}?{}'.format(base_url, query_string)
	return redirect(url)

def get_criteria_values(request):
	feature = request.GET.get('feature')
	category = request.GET.get('category')
	product = request.GET.get('product')
	ins = Criteria.objects.filter(feature=feature).filter(category=category).filter(product=product).first()
	crih_ins = CriteriaHelper.objects.filter(criteria=ins)
	entries, scores = get_criteria_values_helper(crih_ins)
	data = {
		'api': ins.api,
		'data_source': ins.data_source,
		'key': ins.key,
		'entries': entries,
		'scores': scores
	}
	return JsonResponse(data)

def get_criteria_values_helper(crih_ins):
	entries = ""
	scores = ""
	if(len(crih_ins) != 0):
		entries += crih_ins[0].entry
		scores += str(crih_ins[0].score)
		i = 0
		for instance in crih_ins:
			print(len(crih_ins))
			if(i == 0):
				i += 1
				continue
			entries += ','
			scores += ','
			entries += instance.entry
			scores += str(instance.score)
	return entries, scores

def set_scale(request):
	params = get_params(request, 'add4')
	form = SetScaleForm()
	if(params['add'] == 'ok5'):
		messages.info(request, 'Record created successfully')
	context = {'form':form, 'session':params['session']}
	return render(request, 'loan_admin/set_scale.html', context)

def addScale(request):
	form = SetScaleForm(request.POST)
	url = reverse('loan_admin:set_scale')
	if form.is_valid():
		form.save()
		base_url = reverse('loan_admin:set_scale')
		query_string =  urlencode({'add4': 'ok5'})
		url = '{}?{}'.format(base_url, query_string)
	return redirect(url)

def get_scale_values(request):
	ins = SetScale.objects.all().first()
	data = {
		'red': ins.red,
		'green': ins.green,
	}
	return JsonResponse(data)

def uploadCSV(request):
	params = get_params(request, 'add3')
	form = UploadFileForm()
	session = Sessions.objects.all().first()
	if(params['add'] == 'ok4'):
		messages.info(request, 'Record created successfully')
	context = {'form':form, 'session':params['session']}
	return render(request, 'loan_admin/uploadCSV.html', context)

@require_POST
def addApplicant(request):
	form = UploadFileForm(request.POST, request.FILES)
	url = reverse('loan_admin:uploadCSV')
	print(form.errors)
	print(form.is_valid(), file=sys.stderr)
	if form.is_valid():
		form.process_data(request.POST, request.FILES['file'])
		base_url = reverse('loan_admin:uploadCSV')
		query_string =  urlencode({'add3': 'ok4'})
		url = '{}?{}'.format(base_url, query_string)
	return redirect(url)