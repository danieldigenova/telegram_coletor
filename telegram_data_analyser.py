import os  # Operating system interactions
import pandas as pd  # Data analysis and manipulation
import re  # Regular expressions
from urllib.parse import urlparse  # URL parsing
import warnings  # Warning management
warnings.filterwarnings('ignore')  # Ignore warnings
import tldextract  # Domain components extraction from URLs
import matplotlib.pyplot as plt  # Graph creation
import matplotlib.dates as mdates  # Date formatting in graphs
import numpy as np  # Scientific computing
from collections import Counter  # Elements counting
from nltk.tokenize import word_tokenize  # Word tokenization
from nltk.corpus import stopwords  # Stopwords list
import nltk  # Natural Language Toolkit
import itertools  # Iterator operations
from wordcloud import WordCloud  # Word cloud generation
from nltk.stem import WordNetLemmatizer  # Word lemmatization
from nltk import bigrams  # Bigrams generation
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF feature matrix

# Necessary NLTK downloads
nltk.download('stopwords')
nltk.download('rslp')
nltk.download('punkt')
nltk.download('wordnet')


class DataAnalyzer:
    def __init__(self):
        pass  

    def analyze_time_series(self, df):
        # Converting 'date' column to datetime
        df['date'] = pd.to_datetime(df['date'])

        # Determine the collection period
        delta = df['date'].max() - df['date'].min()

        # Define grouping according to the period
        if delta.days < 7:
            freq = 'D'
        elif delta.days == 7:
            freq = 'D'
        elif delta.days <= 15:
            freq = 'D'
        elif delta.days <= 30:
            freq = 'W'
        elif delta.days <= 365:
            freq = 'M'
        else:
            freq = 'A'

        # Grouping according to the determined frequency
        grouped = df.groupby(pd.Grouper(key='date', freq=freq)).size().reset_index(name='counts')

        # Convert dates to numeric for regression, if necessary
        grouped['date_num'] = mdates.date2num(grouped['date'])

        # Linear regression
        z = np.polyfit(grouped['date_num'], grouped['counts'], 1)
        p = np.poly1d(z)

        # Plotting
        plt.figure(figsize=(14, 7))
        plt.plot(grouped['date'], grouped['counts'], label='Mentions', marker='o', linestyle='-', color='skyblue')
        plt.plot(grouped['date'], p(grouped['date_num']), label='Trend Line', color='red', linestyle='--')

        # Formatting the graph
        plt.title('Posts by Period')
        plt.xlabel('Date')
        plt.ylabel('Number of mentions')
        plt.legend()
        plt.grid(True)

        # Adjustments on the x-axis according to the period
        if freq == 'D':
            formatter = mdates.DateFormatter('%d %b')
            locator = mdates.DayLocator()
        elif freq == 'W':
            formatter = mdates.DateFormatter('%d %b')
            locator = mdates.WeekdayLocator()
        elif freq == 'M':
            formatter = mdates.DateFormatter('%b %Y')
            locator = mdates.MonthLocator()
        else:
            formatter = mdates.DateFormatter('%Y')
            locator = mdates.YearLocator()

        plt.gca().xaxis.set_major_formatter(formatter)
        plt.gca().xaxis.set_major_locator(locator)
        plt.gcf().autofmt_xdate()

        plt.show()
        
    def analyze_channels_stats(self, df):
        """Analyze statistics of channels from a DataFrame."""
        # Group data by 'channel_title' and aggregate the number of posts and total views
        channel_summary = df.groupby('channel_title').agg(
            number_of_posts=pd.NamedAgg(column='id', aggfunc='count'), 
            total_views=pd.NamedAgg(column='views', aggfunc='sum')
            ).reset_index()

        # Convert 'total_views' to integers for clean formatting
        channel_summary['total_views'] = channel_summary['total_views'].astype(int)

        # Rename the columns for clarity and better understanding
        channel_summary_renamed = channel_summary.rename(columns={
            'channel_title': 'Channel',
            'number_of_posts': 'Posts',
            'total_views': 'Views'
        })

        # Return the summary table instead of printing it
        return channel_summary_renamed
        
    @staticmethod
    def extract_hashtags(s):
        """Static method to extract hashtags from a string."""
        if isinstance(s, str):
            return re.findall(r'#\w+', s)  # Use regex to find all hashtags in the string
        else:
            return []  # Return an empty list if not a string
        
    def count_hashtags(self, df, text_column='text'):
        """Method to count hashtags in a DataFrame column."""
        if text_column not in df.columns:
            raise KeyError(f"The column '{text_column}' does not exist in the DataFrame.")
        
        # Apply the static method to extract hashtags and expand the resulting list
        hashtags_series = df[text_column].apply(self.extract_hashtags).explode()

        # Count the frequency of each hashtag
        hashtags_count = hashtags_series.value_counts().reset_index()
        hashtags_count.columns = ['Hashtag', 'Posts']

        # Return the resulting DataFrame
        return hashtags_count
    