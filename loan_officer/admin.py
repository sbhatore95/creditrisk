from django.contrib import admin

# Register your models here.
from .models import ModelSchema
admin.site.register(ModelSchema)