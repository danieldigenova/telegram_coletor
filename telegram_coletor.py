import os
from telethon.sync import TelegramClient                          # Permite a comunicação síncrona com a API do Telegram.
from telethon.tl.functions.channels import GetFullChannelRequest  # Permite a coleta de informações do canal
from telethon.errors import ApiIdInvalidError                     # Biblioteca para tratamento de erros de autenticação
from urlextract import URLExtract                                 # Biblioteca para extração de URLs
import asyncio                                                    # Biblioteca para escrever código assíncrono em Python.
import nest_asyncio                                               # Torna possível usar loops asyncio aninhados.
import datetime                                                   # Fornece classes para manipulação de datas e horas.
import time                                                       # Contém funções de tempo, como delay (sleep).
import json                                                       # Facilita a codificação e decodificação de dados em formato JSON.
import pytz                                                       # Oferece suporte para fusos horários.
import re                                                         # Fornece operações com expressões regulares.
import pandas as pd                                               # Biblioteca para manipulação e análise de dados.
import sqlite3                                                    # Biblioteca de banco de dados para armazenar os dados de sessão
from tqdm.auto import tqdm                                        # Permite a exibição de uma barra de progresso

nest_asyncio.apply() # Aplica um patch para permitir que loops asyncio aninhados funcionem corretamente.

# função que coleta as mensagens e salva em um arquivo json
async def coletar(name, api_id, api_hash, filename, termos, canais, limite, data_inicio, data_fim):
    tz = pytz.timezone('GMT')

    # verifica se a data de início está no formato correto
    try:
        data_i = datetime.datetime.strptime(data_inicio, '%d/%m/%y').replace(tzinfo=tz)
    except ValueError:
        print("Erro: Formato de data de início inválido, o formato deve seguir o padrão dd/mm/yy")
        return;

    # verifica se a data final está no formato correto
    try:
        data_f = datetime.datetime.strptime(data_fim, '%d/%m/%y').replace(tzinfo=tz)
        data_f = data_f.replace(hour=23, minute=59)
    except ValueError:
        print("Erro: Formato de data final inválido, o formato deve seguir o padrão dd/mm/yy")
        return;

    # verifica se a lista de canais é uma lista de strings
    if not all(isinstance(s, str) for s in canais):
      print("Erro: Lista de canais inválida. Verifique se todos os nomes de canais são strings.")
      return;

    # verifica se a lista de termos é uma lista de strings
    if not all(isinstance(s, str) for s in termos):
      print("Erro: Lista de termos inválida. Verifique se todos os termos são strings.")
      return;

    try:
      with open(filename + '.json', 'w+') as json_file:
          async with TelegramClient(name, api_id, api_hash) as client:
              pbar = tqdm(canais)
              for canal in pbar:
                  first = 1
                  pbar.set_description("Coletando as mensagens do canal %s" % canal)

                  # Verifica se o canal existe
                  try:
                    messages = await client.get_messages(canal, limit=limite, offset_date = data_f)
                  except ValueError:
                    print("Canal "+ canal + " não encontrado")
                    continue
                  except :
                    print("Não foi possível encontrar o canal " + canal)
                    continue

                  for message in messages:

                      # encerra se a mensagem for mais antiga que a data inicial estabelecida
                      if(message.date < data_i):
                        break

                      if(first):
                        first = 0
                        # recebe as informações do canal
                        channel = await client.get_entity(message.peer_id)

                        channel_full_info = await client(GetFullChannelRequest(channel=channel))
                        participants_count = channel_full_info.full_chat.participants_count

                        group_type = ''

                        if(channel_full_info.chats[0].megagroup):
                            group_type = 'megagroup'
                        elif(channel_full_info.chats[0].broadcast):
                            group_type = 'broadcast'
                        elif(channel_full_info.chats[0].gigagroup):
                            group_type = 'gigagroup'

                      # verifica qual é o tipo de usuário
                      if(message.from_id != None):
                          try:
                              user = await client.get_entity(message.from_id)
                              user_id = user.id

                              if(hasattr(user, "first_name")):
                                  channel_username = None
                                  user_username = user.username
                                  first_name = user.first_name
                                  last_name = user.last_name
                              else:
                                  channel_username = user.username
                                  user_username = None
                                  first_name = None
                                  last_name = None
                          except:
                              user_id = message.from_id
                              channel_username = None
                              user_username = None
                              first_name = None
                              last_name = None
                      else:
                          user_id = None
                          channel_username = None
                          user_username = None
                          first_name = None
                          last_name = None

                      # extrai o número de comentários exibido no botão do Telegram
                      try:
                          comments_text = re.findall("\d+ comments", message.reply_markup.rows[0].buttons[0].text)[0]

                          if(comments_text != ''):
                              comments = re.findall("\d+", message.reply_markup.rows[0].buttons[0].text)[0]
                      except:
                          comments = 0

                      if(comments == 0):
                          if(message.replies == None):
                            replies = 0
                          else:
                            replies = message.replies.replies

                      else:
                          replies = comments

                      # verifica se a mensagem é uma resposta
                      if(message.reply_to != None):
                          reply_to = message.reply_to.reply_to_msg_id
                      else:
                          reply_to = None

                      # verifica se a mensagem é um encaminhamento
                      fwd_from = ''
                      fwd_from_id = ''
                      fwd_from_name = ''
                      fwd_from_post_id = ''

                      if(message.fwd_from != None):
                        if(hasattr(message.fwd_from.from_id, "channel_id")):
                          fwd_from_id = message.fwd_from.from_id.channel_id
                          fwd_from_post_id = message.fwd_from.channel_post
                          fwd_from = 'channel'
                          fwd_from_name = message.fwd_from.from_name
                        elif(hasattr(message.fwd_from.from_id, "user_id")):
                          fwd_from_id = message.fwd_from.from_id.user_id
                          fwd_from_post_id = message.fwd_from.channel_post
                          fwd_from = 'user'
                          fwd_from_name = message.fwd_from.from_name
                        else:
                          fwd_from_id = message.fwd_from.from_id
                          fwd_from_post_id = message.fwd_from.channel_post
                          fwd_from_name = message.fwd_from.from_name

                      # extrai links presentes na mensagem
                      extractor = URLExtract()
                      try:
                        links = extractor.find_urls(message.message)
                      except:
                        links = []

                      # extrai menções presentes na mensagem
                      try:
                        mentions = re.findall("(?<!\S)@(\w+)", message.message)
                      except:
                        mentions = []

                      # constroi link direto para a mensagem
                      msg_link = "https://t.me/" + canal + "/" + str(message.id);

                      # constrói o json e salva em um arquivo
                      json_tweet = {    "id":message.id, "date":str(message.date), "text":message.message,
                                        "post_author":message.post_author, "reply_to_message_id":reply_to,
                                        "views":message.views, "replies":replies, "forwards":message.forwards,
                                        "mentions":mentions, "links":links, "msg_link": msg_link,
                                        "fwd_from":fwd_from, "fwd_from_name":fwd_from_name,
                                        "forward_from_id":fwd_from_id, "forward_from_post_id":fwd_from_post_id,
                                        "channel_title":channel.title, "channel_username":channel.username, "group_type":group_type,
                                        "channel_id": message.peer_id.channel_id, "participants_count": participants_count,
                                        "user_id":user_id, "user_username":user_username, "user_first_name":first_name,
                                        "user_last_name":last_name, "channel_username_chat":channel_username}

                      json.dump(json_tweet, json_file)
                      json_file.write('\n')
                      time.sleep(0.13) # evita que o número máximo de requisições seja atingido

                  json_file.flush()
                  print("Canal " + canal + " coletado.")
              client.disconnect()

    # avisa se já possui uma sessão ativa
    except sqlite3.OperationalError:
        print("Erro: É necessário se autenticar novamente. Execute novamente o código.")
        if f"{name}.session" in os.listdir():
          os.remove(f"{name}.session")
    # avisa se a autenticação na API está errada
    except ApiIdInvalidError:
      print("Erro: Dados de autenticação inválidos. Verifique seus dados da API.")
