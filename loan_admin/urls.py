from django.urls import path, re_path
from . import views
from django.conf.urls.static import static
from django.conf import settings
from django.conf.urls import url

app_name = 'loan_admin'
urlpatterns = [
	path('', views.index, name='index'),
	path('add', views.addFeature, name='add'),
	path('index/?add=ok1', views.index, name='index1'),
	path('configuration', views.configuration, name='configuration'),
	path('criteria', views.criteria, name='criteria'),
	path('weigh', views.addConfiguration, name='weigh'),
	path('addCriteria', views.addCriteria, name='cri'),
	path('configuration/?add1=ok2', views.configuration, name='configuration1'),
	path('criteria/?add2=ok3', views.criteria, name='criteria1'),
	path('uploadCSV', views.uploadCSV, name='uploadCSV'),
	path('uploadCSV/?add3=ok4', views.uploadCSV, name='uploadCSV1'),
	path('addApplicant', views.addApplicant, name='addApplicant'),
	re_path(r'^ajax/get_feature_values/$', views.get_feature_values, name='get_feature_values'),
	re_path(r'^ajax/get_configuration_values/$', views.get_configuration_values, name='get_configuration_values'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)