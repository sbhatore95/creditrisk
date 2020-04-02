from django.urls import path
from . import views
from django.conf.urls.static import static
from django.conf import settings

app_name = 'loan_officer'
urlpatterns = [
	path('', views.index, name='index'),
	path('uploadCSV', views.uploadCSV, name='uploadCSV'),
	path('result', views.result, name='result'),
	path('add', views.addApplicant, name='add'),
	path('uploadCSV/?add=ok', views.uploadCSV, name='uploadCSV'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)