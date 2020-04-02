from django.contrib import admin

# Register your models here.
from .models import Feature
from .models import Configuration
from .models import ModelSchema

admin.site.register(ModelSchema)
admin.site.register(Feature)
admin.site.register(Configuration)