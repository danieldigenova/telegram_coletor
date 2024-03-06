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