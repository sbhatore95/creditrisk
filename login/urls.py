from django.urls import path
from . import views
from django.conf.urls.static import static
from django.conf import settings

app_name = 'login'
urlpatterns = [
	path('', views.index, name='index'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)