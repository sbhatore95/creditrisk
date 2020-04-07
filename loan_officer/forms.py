import csv
import io
from django import forms
from django.forms import ModelForm
from django.db import connection
from django.db import models
import sys

class FileUploadForm(forms.Form):
    file = forms.FileField(widget=forms.FileInput(attrs={'class': 'custom-file-input', 
        'id': "customFile", 'name': 'filename'}))
	
    def process_data(self, f):
        with open('media/credit_risk/dataset/test_id_dataset.csv', 'wb+') as destination:
            for chunk in f.chunks():
                destination.write(chunk)
            destination.close()

class MyForm(forms.Form):
    loan_id = forms.CharField(max_length=100,
        widget=forms.TextInput(attrs={'class': 'form-control'}))
    rule_based = forms.BooleanField(required=False,
        widget=forms.CheckboxInput(attrs={'class': 'custom-control-input', 'id': "customCheck1"}))
    statistical_based = forms.BooleanField(required=False,
        widget=forms.CheckboxInput(attrs={'class': 'custom-control-input', 'id': "customCheck2"}))
    ML_based = forms.BooleanField(required=False,
        widget=forms.CheckboxInput(attrs={'class': 'custom-control-input', 'id': "customCheck3"}))