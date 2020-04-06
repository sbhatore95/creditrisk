from django.db import models

class SavedState(models.Model):
	stat = models.CharField(max_length=10)
	ml = models.CharField(max_length=10)
	statandml = models.CharField(max_length=10, default="false")