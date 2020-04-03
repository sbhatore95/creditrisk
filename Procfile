release: python3 manage.py makemigrations
release: python3 manage.py migrate
web: gunicorn credit_risk.wsgi
release: python3.7 manage.py process_tasks
