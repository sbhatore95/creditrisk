from django import forms
from .models import Feature
from .models import Configuration
from .models import UploadFile, Criteria, CriteriaHelper
import csv
import io
from bootstrap4.widgets import RadioSelectButtonGroup
from crispy_forms.helper import FormHelper
from loan_officer.models import SavedState
from loan_officer.project import *
from .tasks import *

class FeatureForm(forms.ModelForm):
	class Meta:
		model = Feature
		fields = ['name', 'value', 'data_type', 'category']

class ConfigurationForm(forms.ModelForm):	
	def __init__(self, *args, **kwargs):
		super(ConfigurationForm, self).__init__(*args, **kwargs)
		x = Feature.objects.values('name')
		print(x)
		FEATURE_CHOICES = []
		a = '1'
		for i in x:
			b = (a, i['name'])
			FEATURE_CHOICES.append(b)
			a = a + '1'
		self.fields['feature'] = forms.ChoiceField(choices=FEATURE_CHOICES, required=True )
	class Meta:
		model = Configuration
		fields = '__all__'

class UploadFileForm(forms.ModelForm):
	file = forms.FileField()
	columns = forms.CharField( widget=forms.Textarea )
	nominal_features = forms.CharField( widget=forms.Textarea )
	class Meta:
		model = UploadFile
		fields = '__all__'
	def process_data(self, dict, f):
		if(UploadFile.objects.all().first() == None):
			m = UploadFile(file=f, columns=dict['columns'], nominal_features=dict['nominal_features'])
			m.save()
		else:
			m = UploadFile.objects.all().first()
			m.file = f
			m.columns = dict['columns']
			m.nominal_features = dict['nominal_features']
			m.save()
		cols = UploadFile.objects.all().first().columns.split(',')
		del cols[0]
		noml = UploadFile.objects.all().first().nominal_features.split(',')
		fw = open('dataset.csv', 'w')
		# with open('train_id_dataset.csv', 'wb+') as destination:
		# 	for chunk in f.chunks():
		# 		destination.write(chunk)
		# 	destination.close()
		# f.seek(0)
		f.seek(0)
		line = f.readline()
		count = 0
		while(line != b''):
			line = line.decode('ASCII')
			flag = 0
			lout = ""
			print(line)
			for i in range(0, len(line)):
				if(flag == 1):
					lout = lout + line[i]
				if(line[i] == ','):
					flag = 1
			fw.write(lout)
			line = f.readline()
			count = count + 1
			# print(line)
		fw.close()
		if(SavedState.objects.all().first() == None):
			pass
		else:
			SavedState.objects.all().first().delete()
		print("calling bg task")
		bg_task(cols, noml, "dataset.csv")
		print("return")

class CriteriaForm(forms.ModelForm):
	CATEGORY_CHOICES = [
		('In', 'Individual'),
		('Co', 'Company'),
		('Cy', 'Country'),
	]
	PRODUCT_CHOICES = [
		('Ag', 'Agricultural'),
		('Ho', 'Home'),
		('Pe', 'Personal'),
		('Ve', 'Vehicle'),
	]
	DATA_CHOICES = [
		('xm', 'XML'),
		('js', 'JSON'),
		('sq', 'SQL'),
	]
	category = forms.ChoiceField(label="Category",choices=CATEGORY_CHOICES)
	product = forms.ChoiceField(label="Product",choices=PRODUCT_CHOICES)
	data_source = forms.ChoiceField(
        help_text="Select the data source",
        required=True,
        label="Order Type:",
        widget=RadioSelectButtonGroup,
        choices=((1, 'XML'), (2, 'JSON'), (3, 'SQL')),
        initial=1,
    )
	api = forms.CharField(required=True, label="API/SQL")
	key = forms.CharField(required=False, label="Key")
	
	def __init__(self, *args, **kwargs):
		self.entry_count = 0
		super(CriteriaForm, self).__init__(*args, **kwargs)
		x = Feature.objects.values('name')
		FEATURE_CHOICES = []
		a = '1'
		for field in x:
			b = (a, field['name'])
			FEATURE_CHOICES.append(b)
			a = a + '1'
		self.fields['feature'] = forms.ChoiceField(choices=FEATURE_CHOICES, required=True )
		criterias = CriteriaHelper.objects.filter(
			criteria=self.instance
		)
		i = 0
		for i in range(0, len(criterias)):
			field_name = 'entry_' % (i,)
			f = 'score_' + str(i+1)
			str1 = "Criteria " + str(i+1)
			str2 = "Score " + str(i+1)
			self.entry_count += 1
			self.fields[field_name] = forms.CharField(label=str1,required=False)
			self.fields[f] = forms.CharField(label=str2,required=False)
			try:
				self.initial[field_name] = criterias[i].entry
				self.initial[f] = criterias[i].score
			except IndexError:
				self.initial[field_name] = ""
				self.initial[f] = ""
		field_name = 'entry_' + str(i+1)
		f = 'score_' + str(i+1)
		str1 = "Criteria " + str(i+1)
		str2 = "Score " + str(i+1)
		self.entry_count += 1
		self.fields[field_name] = forms.CharField(label=str1,required=False, 
			widget=forms.TextInput(attrs={'id':'new_entry'}))
		self.fields[f] = forms.CharField(label=str2,required=False, 
			widget=forms.TextInput(attrs={'id':'new_score'}))
		count = 1
		if(args):
			print(args[0])
			for key in args[0]:
				if(key[:5] == "entry"):
					count += 1
			print("count " + str(count))
			for i in range(1, count-1):
				field_name = 'entry_' + str(i+1)
				f = 'score_' + str(i+1)
				str1 = "Criteria " + str(i+1)
				str2 = "Score " + str(i+1)
				self.entry_count += 1
				self.fields[field_name] = forms.CharField(label=str1,required=False, 
					widget=forms.TextInput(attrs={'id':'new_entry'}))
				self.fields[f] = forms.CharField(label=str2,required=False, 
					widget=forms.TextInput(attrs={'id':'new_score'}))

	def clean(self):
		entries = []
		i = 1
		field_name = 'entry_%s' % (i,)
		print("<>")
		print(self.cleaned_data)
		print("<>")
		while self.cleaned_data.get(field_name):
			# print(dict(field_name))
			entry = self.cleaned_data[field_name]
			if entry in entries:
				self.add_error(field_name, 'Duplicate')
			else:
				entries.append(entry)
			i += 1
			field_name = 'entry_%s' % (i,)
		self.cleaned_data["entries"] = entries
		scores = []
		i = 1
		f = 'score_%s' % (i,)
		while self.cleaned_data.get(f):
			score = self.cleaned_data[f]
			if score in scores:
				self.add_error(f, 'Duplicate')
			else:
				scores.append(score)
			i += 1
			f = 'score_%s' % (i,)
		self.cleaned_data["scores"] = scores

	def save(self):
		criteria = self.instance
		criteria.feature = self.cleaned_data["feature"]
		criteria.category = self.cleaned_data["category"]
		criteria.api = self.cleaned_data["api"]
		criteria.key = self.cleaned_data["key"]
		criteria.data_source = self.cleaned_data["data_source"]
		criteria.product = self.cleaned_data["product"]
		criteria.save()
		print("-->")
		print(self.cleaned_data["entries"])
		print("<--")
		for i in range(0, len(self.cleaned_data["entries"])):
			CriteriaHelper.objects.create(
				criteria=criteria,
				entry=self.cleaned_data["entries"][i],
				score=self.cleaned_data["scores"][i]
			)
    
	def get_entries(self):
		for field_name in self.fields:
			if field_name.startswith('entry_'):
				yield self[field_name]

	def get_scores(self):
		for field_name in self.fields:
			if field_name.startswith('score_'):
				yield self[field_name]

	# def get_both(self):
	# 	ans = []
	# 	A = []
	# 	B = []
	# 	for field_name in self.fields:
	# 		if field_name.startswith('entry_'):
	# 			A.append(self[field_name])
	# 	for field_name in self.fields:
	# 		if field_name.startswith('score_'):
	# 			B.append(self[field_name])
	# 	for i in range(0, len(A)):
	# 		ans.append(A[i])
	# 		if(i+1 < len(A)):
	# 			ans.append(B[i+1])
	# 	return ans

	class Meta:
		model = Criteria
		fields = ['feature','category','product','data_source','api','key']