from django.shortcuts import render, redirect
from .forms import UserForm
from django.contrib.auth import authenticate
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.urls import reverse
from urllib.parse import urlencode
from django.contrib import messages
from .models import Sessions

def index(request):
	return render(request, 'login/index.html')

def login(request):
	add = request.GET.get('add')
	form = UserForm()
	if(add):
		messages.info(request, 'Password incorrect')
	context = {'form':form}
	return render(request, 'login/login.html', context)

@require_POST
def authenticate_and_redirect(request):
	arr = ['', 'Admin', 'Officer']
	print(request.POST)
	user = authenticate(username=arr[int(request.POST['name'])], password=request.POST['password'])
	if user is not None:
		name = request.POST['name']
		ins = Sessions.objects.all().first()
		if(ins is None):
			new = Sessions(user=name)
			new.save()
		else:
			ins.user = name
			ins.save()
		if name is '1':
			return redirect(reverse('loan_admin:index'))
		if name is '2':
			return redirect('loan_officer:index')
	else:
		base_url = reverse('login:login')
		query_string =  urlencode({'add': 'invalid'})
		url = '{}?{}'.format(base_url, query_string)
		return redirect(url)

def logout(request):
	ins = Sessions.objects.all().first()
	ins.user = "None"
	ins.save()
	return redirect('login:login')
