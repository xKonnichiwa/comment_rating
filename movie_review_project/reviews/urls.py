from django.urls import path
from .views import review_view

urlpatterns = [
    path('', review_view, name='review'),
]
