from django import forms
from bootstrap4.widgets import RadioSelectButtonGroup

class UserForm(forms.Form):
	name = forms.ChoiceField(
			help_text="Select the User",
			required=True,
			label="Select User ",
			widget=RadioSelectButtonGroup(attrs={'class': 'form-control'}),
			choices=((1, 'Admin'), (2, 'Officer')),
			initial=1,
			)
	password = forms.CharField(label="Password", required=True, 
		widget=forms.PasswordInput(attrs={'class': 'form-control'}))