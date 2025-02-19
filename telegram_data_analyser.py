#atualizado em 19 de fevereiro de 2025

import os  # Interações com o sistema operacional
import re  # Expressões regulares
from urllib.parse import urlparse  # Análise de URLs
import warnings  # Gerenciamento de avisos
warnings.filterwarnings('ignore')  # Ignorar avisos

import pandas as pd  # Análise e manipulação de dados
import numpy as np  # Computação científica
import matplotlib.pyplot as plt  # Criação de gráficos
import matplotlib.dates as mdates  # Formatação de datas nos gráficos
import matplotlib.ticker as ticker
import tldextract  # Extração de componentes de domínios de URLs
from collections import Counter  # Contagem de elementos

from nltk.tokenize import word_tokenize  # Tokenização de palavras
from nltk.corpus import stopwords  # Lista de stopwords
import nltk  # Toolkit de linguagem natural
from nltk import bigrams  # Geração de bigramas
from nltk.stem import WordNetLemmatizer  # Lematização de palavras

from wordcloud import WordCloud  # Geração de nuvem de palavras
from datetime import timedelta
from typing import List, Any

# Downloads necessários do NLTK
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('rslp')  # Caso utilize em outros processamentos

class DataAnalyzer:
    def __init__(self) -> None:
        self.stop_words = set(stopwords.words('portuguese'))
        self.link_shorteners = [
            "bit.ly", "goo.gl", "tinyurl.com", "t.co", "ow.ly",
            "bitly.com", "rebrand.ly", "cutt.ly", "shorte.st",
            "tiny.cc", "is.gd", "soo.gd", "s2r.co", "clicky.me", "budurl.com"
        ]
        # Pré-compilação de padrões regex para melhor performance
        self.url_pattern = re.compile(r'http[s]?://\S+')
        self.hashtag_pattern = re.compile(r'#\w+')
        self.mention_pattern = re.compile(r'@\S+')
        self.non_alnum_pattern = re.compile(r'[^\w\s]')

    def analyze_time_series(self, df: pd.DataFrame) -> None:
        """
        Plota uma análise de séries temporais das postagens, exibindo apenas a data
        (dia e mês) no eixo X. Exibe no título o intervalo de datas completo (com ano).
        """
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        start_date = df['date'].min()
        end_date = df['date'].max()
        delta = end_date - start_date
    
        # Define a frequência de agrupamento conforme o intervalo de datas
        if delta < timedelta(hours=24):
            # Menor que 1 dia: agrupar por hora, mas exibir apenas dia/mês no eixo
            freq = 'H'
            major_locator = mdates.DayLocator()
            major_formatter = mdates.DateFormatter('%d %b')
            minor_locator = None
            xlabel = 'Data (por hora)'
        elif delta < timedelta(days=7):
            # Menor que 1 semana: agrupar por hora
            freq = 'H'
            major_locator = mdates.DayLocator()
            major_formatter = mdates.DateFormatter('%d %b')
            minor_locator = None
            xlabel = 'Data (por hora)'
        elif delta < timedelta(days=90):
            # Menor que 3 meses: agrupar por dia
            freq = 'D'
            major_locator = mdates.DayLocator(interval=3)
            major_formatter = mdates.DateFormatter('%d %b')
            minor_locator = mdates.DayLocator()
            xlabel = 'Data (por dia)'
        elif delta < timedelta(days=365):
            # Menor que 1 ano: agrupar por semana
            freq = '7D'
            major_locator = mdates.WeekdayLocator(byweekday=mdates.MO)
            major_formatter = mdates.DateFormatter('%d %b')
            minor_locator = mdates.DayLocator()
            xlabel = 'Data (por semana)'
        elif delta < timedelta(days=365 * 2):
            # Entre 1 e 2 anos: agrupar por mês
            freq = 'M'
            major_locator = mdates.MonthLocator()
            major_formatter = mdates.DateFormatter('%b %Y')
            minor_locator = None
            xlabel = 'Data (por mês)'
        else:
            # Acima de 2 anos: agrupar por ano
            freq = 'A'
            major_locator = mdates.YearLocator()
            major_formatter = mdates.DateFormatter('%Y')
            minor_locator = None
            xlabel = 'Data (por ano)'
    
        # Agrupamento dos dados conforme a frequência definida
        grouped = df.groupby(pd.Grouper(key='date', freq=freq)).size().reset_index(name='counts')
        grouped['date_num'] = mdates.date2num(grouped['date'])
    
        # Regressão linear para a linha de tendência (opcional)
        z = np.polyfit(grouped['date_num'], grouped['counts'], 1)
        p = np.poly1d(z)
    
        fig, ax = plt.subplots(figsize=(14, 7))
        # Linha em cinza escuro, sem marcadores
        ax.plot(
            grouped['date'],
            grouped['counts'],
            linestyle='-',
            color='dimgray',
            label='Mensagens'
        )
        # Para exibir a linha de tendência, descomente a linha abaixo:
        ax.plot(grouped['date'], p(grouped['date_num']), label='Linha de Tendência', color='red', linestyle='--')
    
        ax.set_xlim(start_date, end_date)
    
        # ---- INSERINDO AS DATAS NO TÍTULO ----
        # Formata o intervalo completo com dia, mês e ano
        date_range_str = f"{start_date.strftime('%d %b %Y')} - {end_date.strftime('%d %b %Y')}"
        plt.title(f"Envio de mensagens ao longo do período\n({date_range_str})")
        # --------------------------------------
    
        plt.xlabel(xlabel)
        plt.ylabel('Mensagens')
        plt.legend()
    
        # Grid (linhas claras, tracejadas, finas e semitransparentes)
        ax.grid(True, which='both', axis='both',
                color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    
        # Configuração dos marcadores do eixo x
        ax.xaxis.set_major_locator(major_locator)
        ax.xaxis.set_major_formatter(major_formatter)
    
        if minor_locator is not None:
            ax.xaxis.set_minor_locator(minor_locator)
            ax.tick_params(axis='x', which='minor', length=5, width=1)
    
        ax.tick_params(axis='x', which='major', length=10, width=2)
        plt.gcf().autofmt_xdate(rotation=45, ha='right')
    
        # Configuração do eixo y para mostrar apenas valores inteiros
        ax.set_yticks(np.arange(0, grouped['counts'].max() + 10, 10))
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
        plt.show()

    def analyze_channels_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Retorna estatísticas dos canais, agrupando pelo título do canal.
        """
        channel_summary = df.groupby('channel_title', as_index=False).agg(
            Posts=pd.NamedAgg(column='id', aggfunc='count'),
            Views=pd.NamedAgg(column='views', aggfunc='sum')
        )
        channel_summary['Views'] = channel_summary['Views'].astype(int)
        channel_summary.rename(columns={'channel_title': 'Channel'}, inplace=True)
        return channel_summary

    @staticmethod
    def extract_hashtags(s: Any) -> List[str]:
        """
        Extrai hashtags de uma string.
        """
        if isinstance(s, str):
            return re.findall(r'#\w+', s)
        return []

    def count_hashtags(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """
        Conta a frequência de hashtags em uma coluna do DataFrame.
        """
        if text_column not in df.columns:
            raise KeyError(f"A coluna '{text_column}' não existe no DataFrame.")
        hashtags_series = df[text_column].apply(self.extract_hashtags).explode()
        hashtags_count = hashtags_series.value_counts().reset_index()
        hashtags_count.columns = ['Hashtag', 'Posts']
        return hashtags_count

    def preprocess_text(self, text: Any) -> str:
        """
        Realiza a limpeza do texto removendo URLs, hashtags, menções e pontuação,
        convertendo para minúsculas, tokenizando e extraindo bigramas.
        """
        text = str(text)
        text = self.url_pattern.sub('', text)
        text = self.hashtag_pattern.sub('', text)
        text = self.mention_pattern.sub('', text)
        text = self.non_alnum_pattern.sub('', text)
        text = text.lower()

        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in self.stop_words and word.isalpha()]
        bigram_list = ['_'.join(bg) for bg in bigrams(tokens)]
        return ' '.join(bigram_list)

    def generate_bigrams(self, df: pd.DataFrame, text_column: str) -> None:
        """
        Gera bigramas a partir de uma coluna de texto e exibe uma nuvem de palavras com fundo escuro.
        """
        df = df.copy()
        df['bigrams'] = df[text_column].apply(self.preprocess_text)
        all_bigrams = ' '.join(df['bigrams'].tolist())
        bigram_frequency = Counter(all_bigrams.split())
    
        # Exemplo usando fundo preto e a paleta "inferno"
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='black',  
            colormap='PRGn'         
        ).generate_from_frequencies(bigram_frequency)
    
        plt.figure(figsize=(15, 10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()

    def extract_links(self, text: Any) -> List[str]:
        """
        Extrai URLs de uma string.
        """
        if pd.isna(text):
            return []
        return self.url_pattern.findall(str(text))

    def extract_domain_and_sub(self, url: str) -> str:
        """
        Extrai e formata o domínio e subdomínio de uma URL.
        """
        extracted = tldextract.extract(url)
        domain = extracted.domain.lower()
        subdomain = extracted.subdomain.lower()
        if subdomain:
            subdomain = re.sub(r'www\d*\.', '', subdomain)
        combined = f"{subdomain}.{domain}" if subdomain else domain
        combined = combined.replace("www.", "")

        # Tratamento de casos específicos
        if domain in ("youtube", "youtu") and (extracted.suffix == "be" or domain == "youtube"):
            combined = "youtube"
        if (domain == "twitter" and extracted.suffix == "com") or (domain == "x" and extracted.suffix == "com"):
            combined = "twitter"

        full_domain = f"{combined}.{extracted.suffix}" if extracted.suffix else combined
        if any(shortener in full_domain for shortener in self.link_shorteners):
            return "encurtador de link"
        if domain == "t" and extracted.suffix == "me":
            return "telegram"
        return combined

    def count_domain_frequency(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
        Conta a frequência de domínios extraídos dos links de uma coluna do DataFrame.
        """
        df = df.copy()
        df['extracted_links'] = df[text_column].apply(self.extract_links)
        links_series = df['extracted_links'].explode().dropna()
        main_domains_series = links_series.apply(self.extract_domain_and_sub)
        domain_counts = main_domains_series.value_counts().reset_index()
        domain_counts.columns = ['Domain', 'Posts']
        return domain_counts[domain_counts['Domain'] != '']

    def count_links_and_make_clickable(self, df: pd.DataFrame, text_column: str):
        """
        Conta a frequência de cada link e formata os links para que fiquem clicáveis.
        """
        links_series = df[text_column].apply(self.extract_links).explode()
        links_count = links_series.value_counts().reset_index()
        links_count.columns = ['Link', 'Posts']
        links_styled = links_count.style.format({'Link': lambda x: f'<a href="{x}">{x}</a>'})
        return links_styled

    def messages_with_most_views(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Retorna as mensagens com o maior número de visualizações.
        """
        df = df.copy()
        df_filtered = df[df['views'] > 0]
        if df_filtered.empty:
            print("A tabela está vazia porque não houve visualizações.")
            return df_filtered

        necessary_columns = ['channel_title', 'date', 'text', 'views']
        df_summary = df_filtered[necessary_columns].copy()
        df_summary.rename(columns={
            'channel_title': 'Canal',
            'views': 'Visualizações',
            'text': 'Mensagem',
            'date': 'Data'
        }, inplace=True)
        df_summary.sort_values(by='Visualizações', ascending=False, inplace=True)
        return df_summary

    def create_summary_table_with_forwards(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Retorna uma tabela resumo das mensagens que foram encaminhadas.
        """
        df = df.copy()
        df_filtered = df[df['forwards'] > 0]
        if df_filtered.empty:
            print("A tabela está vazia porque não houve encaminhamentos.")
            return df_filtered

        necessary_columns = ['channel_title', 'date', 'text', 'forwards']
        df_summary = df_filtered[necessary_columns].copy()
        df_summary.rename(columns={
            'forwards': 'Encaminhamentos',
            'channel_title': 'Canal',
            'text': 'Mensagem',
            'date': 'Data'
        }, inplace=True)
        df_summary.sort_values(by='Encaminhamentos', ascending=False, inplace=True)
        return df_summary

    def most_commented_messages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Retorna as mensagens com o maior número de comentários.
        """
        df = df.copy()
        df['replies'] = pd.to_numeric(df['replies'], errors='coerce')
        df_filtered = df[df['replies'] > 0]
        if df_filtered.empty:
            print("A tabela está vazia porque não foi coletada nenhuma resposta.")
            return df_filtered

        # Incluindo a coluna 'date' para manter consistência com os outros resumos
        necessary_columns = ['channel_title', 'date', 'text', 'replies']
        df_summary = df_filtered[necessary_columns].copy()
        df_summary.rename(columns={
            'replies': 'Comentários',
            'channel_title': 'Canal',
            'text': 'Mensagem',
            'date': 'Data'
        }, inplace=True)
        df_summary.sort_values(by='Comentários', ascending=False, inplace=True)
        return df_summary
    
    def combined_summary_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Retorna um único DataFrame com as colunas renomeadas para:
        [Canal, Data, Mensagem, Visualizações, Encaminhamentos, Comentários, Link].
        
        Preenche valores nulos por 0 nas métricas, ordena as linhas pela soma das
        métricas (em ordem decrescente) e reposiciona a coluna 'Link' para o final.
        """
    
        # Copia o df para não alterar o original
        df = df.copy()
    
        # Converte 'replies' para numérico se existir; caso contrário, cria a coluna
        if 'replies' in df.columns:
            df['replies'] = pd.to_numeric(df['replies'], errors='coerce').fillna(0)
        else:
            df['replies'] = 0
    
        # Mapeia os nomes das colunas para os nomes desejados
        rename_map = {
            'channel_title': 'Canal',
            'date': 'Data',
            'text': 'Mensagem',
            'views': 'Visualizações',
            'forwards': 'Encaminhamentos',
            'replies': 'Comentários',
            'msg_link': 'Link'
        }
        # Remove chaves que não existem no df
        for old_col in list(rename_map.keys()):
            if old_col not in df.columns:
                rename_map.pop(old_col)
    
        df.rename(columns=rename_map, inplace=True)
    
        # Preenche valores faltantes com 0 nas métricas
        for col in ['Visualizações', 'Encaminhamentos', 'Comentários']:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(int)
    
        # (Opcional) Ordena as linhas com base na soma das métricas
        df['Total'] = 0
        for col in ['Visualizações', 'Encaminhamentos', 'Comentários']:
            if col in df.columns:
                df['Total'] += df[col]
        df.sort_values(by='Total', ascending=False, inplace=True)
        df.drop(columns='Total', inplace=True)
    
        # Reordena as colunas para que a coluna "Link" fique sempre no final
        if 'Link' in df.columns:
            cols = [col for col in df.columns if col != 'Link'] + ['Link']
            df = df[cols]

        desired_cols = ["Canal", "Data", "Mensagem", "Visualizações", "Encaminhamentos", "Comentários", "Link"]
        existing_cols = [col for col in desired_cols if col in df.columns]
        df = df[existing_cols]

        return df



