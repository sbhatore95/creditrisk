from django.urls import path
from . import views
from django.conf.urls.static import static
from django.conf import settings

app_name = 'login'
urlpatterns = [
	path('', views.index, name='index'),
	path('login', views.login, name='login'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)