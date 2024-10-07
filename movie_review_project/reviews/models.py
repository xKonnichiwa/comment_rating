from django.db import models

class Review(models.Model):
    text = models.TextField(verbose_name='Текст отзыва')
    predicted_rating = models.IntegerField(verbose_name='Предсказанный рейтинг', null=True, blank=True)
    predicted_sentiment = models.CharField(verbose_name='Статус отзыва', max_length=10, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'Отзыв {self.id}'
