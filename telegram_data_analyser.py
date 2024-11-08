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
from datetime import timedelta

# Necessary NLTK downloads
nltk.download('stopwords')
nltk.download('rslp')
nltk.download('punkt')
nltk.download('wordnet')


class DataAnalyzer:
    def __init__(self):
        self.stop_words = set(stopwords.words('portuguese')) 
        self.link_shorteners = [
            "bit.ly", "goo.gl", "tinyurl.com", "t.co", "ow.ly",
            "bitly.com", "rebrand.ly", "cutt.ly", "shorte.st",
            "tiny.cc", "is.gd", "soo.gd", "s2r.co", "clicky.me", "budurl.com"
        ]
    def analyze_time_series(self, df):
        # Converting 'date' column to datetime
        df['date'] = pd.to_datetime(df['date'])

        # Determine the collection period
        delta = df['date'].max() - df['date'].min() - timedelta(days=1)

        # Define grouping according to the period
        if delta.days < 7:
            freq = 'H'
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

        freq = 'D'
        # Grouping according to the determined frequency
        grouped = df.groupby(pd.Grouper(key='date', freq=freq)).size().reset_index(name='counts')

        # Convert dates to numeric for regression
        grouped['date_num'] = mdates.date2num(grouped['date'])

        # Linear regression
        z = np.polyfit(grouped['date_num'], grouped['counts'], 1)
        p = np.poly1d(z)

        # Plotting
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(grouped['date'], grouped['counts'], label='Postagens', marker='o', linestyle='-', color='skyblue')
        #ax.plot(grouped['date'], p(grouped['date_num']), label='Linha de Tendência', color='red', linestyle='--')

        # Ensure the plot starts at the minimum date and ends at the maximum date
        ax.set_xlim([df['date'].min(), df['date'].max()])

        # Formatting the graph
        plt.title('Postagens por Período')
        plt.xlabel('Data e Hora' if delta.days < 3 else 'Data')
        plt.ylabel('Número de Postagens')
        plt.legend()
        plt.grid(True, which='both', axis='x')

        # Adjustments on the x-axis for dates and hours
        if delta.days < 3:
            major_locator_x = mdates.DayLocator()
            major_formatter_x = mdates.DateFormatter('%d %b')
            minor_locator_x = mdates.HourLocator(interval=4)
            minor_formatter_x = mdates.DateFormatter('%H:%M')
        else:
            major_locator_x = mdates.DayLocator()
            major_formatter_x = mdates.DateFormatter('%d %b')
            minor_locator_x = None
            minor_formatter_x = None

        ax.xaxis.set_major_locator(major_locator_x)
        ax.xaxis.set_major_formatter(major_formatter_x)

        if minor_locator_x and minor_formatter_x:
            ax.xaxis.set_minor_locator(minor_locator_x)
            ax.xaxis.set_minor_formatter(minor_formatter_x)
            ax.tick_params(axis='x', which='major', length=10, width=2)
            ax.tick_params(axis='x', which='minor', length=5, width=1)
        else:
            ax.tick_params(axis='x', which='major', length=10, width=2)

        plt.gcf().autofmt_xdate(rotation=45, ha='right')

        # Configure y-axis to show only integer counts
        ax.set_yticks(np.arange(0, grouped['counts'].max() + 10, 10))
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

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
    
    def preprocess_text(self, text):
        """Preprocess text by cleaning and extracting bigrams."""
        # Converte para string, remove URLs, hashtags e menções
        text = str(text)
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'#\S+', '', text)
        text = re.sub(r'@\S+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = text.lower()

        # Tokenizar e remover stopwords
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in self.stop_words and word.isalpha()]

        # Extrair bigramas e retornar como string com bigramas separados por _
        bigram_list = ['_'.join(bg) for bg in bigrams(tokens)]
        return ' '.join(bigram_list)

    def generate_bigrams(self, df, text_column):
        """Generate bigrams for a DataFrame text column and create a word cloud."""
        df['bigrams'] = df[text_column].apply(self.preprocess_text)
        all_bigrams = ' '.join(df['bigrams'].tolist())

        # Contar a frequência dos bigramas
        bigram_frequency = Counter(all_bigrams.split())

        # Gerar a nuvem de palavras com base nos bigramas
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(bigram_frequency)

        # Mostrar a nuvem de palavras
        plt.figure(figsize=(15, 10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()
        
    def extract_links(self, text):
        """Extract URLs from a text string. Ensure text is a string and not NaN."""
        if pd.isna(text):
            return []  # Retorna uma lista vazia para valores NaN
        text = str(text)  # Garante que o texto seja uma string
        return re.findall(r'http[s]?://\S+', text)

    def extract_domain_and_sub(self, url):
        extracted = tldextract.extract(url)
        domain = extracted.domain
        subdomain = re.sub(r'www\d*\.', '', extracted.subdomain)

        combined = f"{subdomain}.{domain}" if subdomain else domain
        combined = combined.replace("www.", "")

        if domain == "youtube" or (domain == "youtu" and extracted.suffix == "be"):
            combined = "youtube"

        if (domain == "twitter" and extracted.suffix == "com") or (domain == "x" and extracted.suffix == "com"):
            combined = "twitter"

        full_domain = f"{combined}.{extracted.suffix}" if extracted.suffix else combined
        if any(shortener in full_domain for shortener in self.link_shorteners):
            combined = "encurtador de link"

        if domain == "t" and extracted.suffix == "me":
            combined = "telegram"

        return combined

    def count_domain_frequency(self, df, text_column):
        df['extracted_links'] = df[text_column].apply(self.extract_links)
        links_series = df['extracted_links'].explode().dropna()
        main_domains_series = links_series.apply(self.extract_domain_and_sub)
        domain_counts = main_domains_series.value_counts().reset_index()
        domain_counts.columns = ['Domain', 'Posts']
        domain_counts = domain_counts[domain_counts['Domain'] != '']
        
        return domain_counts
    
    def count_links_and_make_clickable(self, df, text_column):
        """Count the frequency of each link and format links to be clickable in a DataFrame."""
        links_series = df[text_column].apply(self.extract_links).explode()
        links_count = links_series.value_counts().reset_index()
        links_count.columns = ['Link', 'Posts']

        # Apply formatting to make links clickable
        links_styled = links_count.style.format({'Link': lambda x: f'<a href="{x}">{x}</a>'})
        
        return links_styled
    
    def messages_with_most_views(self, df):
        # Filters the DataFrame to only include rows where the number of views is greater than 0
        df_filtered = df[df['views'] > 0]
        
        if df_filtered.empty:
            print("A tabela está vazia porque não houve visualizações.")
            return df_filtered  # Returns the empty DataFrame and the message has already been displayed
        
        # Extracts the necessary columns from the filtered DataFrame
        necessary_columns = ['channel_title', 'date', 'text', 'views', ]
        df_summary = df_filtered[necessary_columns].copy()
        
        # Renames the columns
        df_summary.rename(columns={'channel_title': 'Canal', 
                                'views': 'Visualizações',
                                'text': 'Mensagem',
                                'date': 'Data'
                                }, 
                        inplace=True)
        
        # Sorts the DataFrame in descending order of views
        df_summary.sort_values(by='Visualizações', ascending=False, inplace=True)
        
        return df_summary
        
    def create_summary_table_with_forwards(self, df):
        # Filters the DataFrame to only include rows where the number of forwards is greater than 0
        df_filtered = df[df['forwards'] > 0]
        
        if df_filtered.empty:
            print("A tabela está vazia porque não houve encaminhamentos.")
            return df_filtered  # Returns the empty DataFrame and the message has already been displayed
        
        # Extracts the necessary columns from the filtered DataFrame
        necessary_columns = ['channel_title', 'date', 'text', 'forwards']
        df_summary = df_filtered[necessary_columns].copy()
        
        # Renames the 'forwards' column to 'Total Forwards'
        df_summary.rename(columns={'forwards': 'Encaminhamentos', 
                                'channel_title': 'Canal',
                                'text': 'Mensagem',
                                'date': 'Data'
                                }, 
                        inplace=True)
        
        # Sorts the DataFrame in descending order of fowards
        df_summary.sort_values(by='Encaminhamentos', ascending=False, inplace=True)
        
        return df_summary
    
    def most_commented_messages(self, df):
        # Ensures 'replies' is treated as numeric, if not already.
        df['replies'] = pd.to_numeric(df['replies'], errors='coerce')
        
        # Filters to include only messages with comments (replies > 0)
        df_filtered = df[df['replies'] > 0]
        
        # Sorts messages by the number of comments in descending order
        df_sorted = df_filtered.sort_values(by='replies', ascending=False)
        
        # Selects the columns of interest
        necessary_columns = ['channel_title', 'text', 'replies']
        df_summary = df_sorted[necessary_columns].copy()
        
        # Renames the 'replies' column to 'Comentários'
        df_summary.rename(columns={'replies': 'Comentários', 
                        'channel_title': 'Canal',
                        'text': 'Mensagem',
                        'date': 'Data'
                        }, 
                inplace=True)
        
        # Checks if the resulting DataFrame is empty
        if df_summary.empty:
            print("A tabela está vazia porque não foi coletada nenhuma resposta.")
            return df_summary  # You can choose to return here or outside the if
        
        return df_summary
   
    