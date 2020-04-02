import csv
import io
from django import forms
from django.forms import ModelForm
from django.db import connection
from django.db import models
import sys
from .models import ModelSchema, FieldSchema, FileUpload
from bootstrap4.widgets import RadioSelectButtonGroup

# class CSVForm(forms.Form):
# 	class Meta:
# 		model = CSV
# 		fields = '__all__'

# 	data_file = forms.FileField()

# 	def process_data(self):
# 		f = io.TextIOWrapper(self.cleaned_data['data_file'].file)
# 		reader = csv.DictReader(f)
# 		rest = [row for row in reader]
# 		if len(ModelSchema.objects.all()) != 0:
# 			applicant_model_schema = ModelSchema.objects.first()
# 		else:
# 			applicant_model_schema = ModelSchema.objects.create(name='LoanApplicant')
# 		field_schema = FieldSchema.objects.create(name='all_field', data_type='character')
# 		color = applicant_model_schema.add_field(
# 			field_schema,
# 			null = True,
# 		)
# 		LoanApplicant = applicant_model_schema.as_model()
# 		for j in rest:
# 			a = LoanApplicant(all_field=j)
# 			a.save()

class FileUploadForm(ModelForm):
	class Meta:
		model = FileUpload
		fields = '__all__'
	# def process_data(self, dict, f):
	# 	with open('id_dataset.csv', 'wb+') as destination:
	# 		for chunk in f.chunks():
	# 			destination.write(chunk)
	# 		destination.close()

class MyForm(forms.Form):
    # media_type = forms.ChoiceField(
    #     help_text="Select the order type.",
    #     required=True,
    #     label="Order Type:",
    #     widget=RadioSelectButtonGroup,
    #     choices=((1, 'Rule Based'), (2, 'Statistical based'), (3, 'ML based')),
    #     initial=1,
    # )
    loan_id = forms.CharField(max_length=100)
    rule_based = forms.BooleanField(required=False)
    statistical_based = forms.BooleanField(required=False)
    ML_based = forms.BooleanField(required=False)