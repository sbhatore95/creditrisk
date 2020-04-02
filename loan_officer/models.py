from django.db import models

class FileUpload(models.Model):
	file = models.FileField(upload_to='credit_risk/dataset')

class SavedState(models.Model):
	stat = models.CharField(max_length=10)
	ml = models.CharField(max_length=10)
	statandml = models.CharField(max_length=10, default="false")