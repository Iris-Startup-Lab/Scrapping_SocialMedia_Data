# backend/scraping/management/commands/scrape_twitter.py
from django.core.management.base import BaseCommand
#from backend.scraping.scripts.twitter_scraper import get_tweet_details
from scraping.scripts.twitter_scraper import get_tweet_details

class Command(BaseCommand):
    help = 'Scrapes tweets from Twitter (X) based on a tweet ID'
    def add_arguments(self, parser):
        parser.add_argument('tweet_id', type=str, help='The search query for Twitter (X)')
        parser.add_argument('--count', type=int, default=10, help='Number of tweets to scrape')

    def handle(self, *args, **options):
        query = options['tweet_id']
        count = options['count']
        self.stdout.write(self.style.SUCCESS(f'Starting to scrape {count} tweets for tweet id: "{query}"...'))
        get_tweet_details(query, max_results=count)
        self.stdout.write(self.style.SUCCESS('Successfully finished scraping tweets.'))