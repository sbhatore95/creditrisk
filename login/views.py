from django.shortcuts import render

def index(request):
	return render(request, 'login/index.html')

def login(request):
	return render(request, 'login/login.html')