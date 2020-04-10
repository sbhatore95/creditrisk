from django.db import models

class Sessions(models.Model):
	user = models.CharField(max_length=10)

	def destroy(self):
		self.user = "None"
		self.save()