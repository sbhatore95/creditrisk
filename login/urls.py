from django.urls import path, re_path
from . import views
from django.conf.urls.static import static
from django.conf import settings
import os

STATIC_ROOT = os.path.join(settings.BASE_DIR, "static/login")

app_name = 'login'
urlpatterns = [
	path('', views.index, name='index'),
	path('login', views.login, name='login'),
	path('login?add=invalid', views.login, name='login1'),
	path('authenticate_and_redirect', views.authenticate_and_redirect, name='authenticate_and_redirect'),
	path('login/logout', views.logout, name="logout"),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)