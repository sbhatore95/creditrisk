# Generated by Django 3.0.4 on 2020-04-09 09:23

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('loan_admin', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='feature',
            name='status',
            field=models.BooleanField(default=False),
        ),
    ]