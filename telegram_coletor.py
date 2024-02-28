from telethon.sync import TelegramClient                          # Permite a comunicação síncrona com a API do Telegram.
from telethon.tl.functions.channels import GetFullChannelRequest  # Permite a coleta de informações do canal
from urlextract import URLExtract                                 # Biblioteca para extração de URLs
import asyncio                                                    # Biblioteca para escrever código assíncrono em Python.
import nest_asyncio                                               # Torna possível usar loops asyncio aninhados.
import datetime                                                   # Fornece classes para manipulação de datas e horas.
import time                                                       # Contém funções de tempo, como delay (sleep).
import json                                                       # Facilita a codificação e decodificação de dados em formato JSON.
import pytz                                                       # Oferece suporte para fusos horários.
import re                                                         # Fornece operações com expressões regulares.
import pandas as pd                                               # Biblioteca para manipulação e análise de dados.

nest_asyncio.apply() # Aplica um patch para permitir que loops asyncio aninhados funcionem corretamente.

# função que coleta as mensagens e salva em um arquivo json
async def coletar(name, api_id, api_hash, filename, termos, canais, data_inicio, data_fim):
    tz = pytz.timezone('GMT')
    data_i = datetime.datetime.strptime(data_inicio, '%d/%m/%y').replace(tzinfo=tz)
    data_f = datetime.datetime.strptime(data_fim, '%d/%m/%y').replace(tzinfo=tz)

    with open(filename + '.json', 'a+') as json_file:
        async with TelegramClient(name, api_id, api_hash) as client:
            for canal in canais:
                messages = await client.get_messages(canal, limit=3000, offset_date = data_f)
                for message in messages:

                    # encerra se a mensagem for mais antiga que a data inicial estabelecida
                    if(message.date < data_i):
                        break

                    # se a lista de termos for vazia então coleta todas mensagens
                    if(termos == []):
                      check = True
                    # se a lista de termos ter algum termo, então realiza a verificação dos termos
                    else:
                      texto = str(message.message).lower()
                      palavras = texto.split()

                      # verifica se a mensagem possui algum dos termos presentes na lista fornecida
                      check = any(termo in termos for termo in palavras)

                    if(check):

                        # recebe as informações do canal
                        channel = await client.get_entity(message.peer_id)

                        channel_full_info = await client(GetFullChannelRequest(channel=channel))
                        participants_count = channel_full_info.full_chat.participants_count

                        group_type = '';

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
                        links = extractor.find_urls(message.message)

                        # extrai menções presentes na mensagem
                        mentions = re.findall("(?<!\S)@(\w+)", message.message)

                        # constrói o json e salva em um arquivo
                        #try:
                        json_tweet = {    "id":message.id, "date":str(message.date), "text":message.message,
                                          "post_author":message.post_author, "reply_to_message_id":reply_to,
                                          "views":message.views, "replies":replies, "forwards":message.forwards, 
                                          "mentions":mentions, "links":links,
                                          "fwd_from":fwd_from, "fwd_from_name":fwd_from_name,
                                          "forward_from_id":fwd_from_id, "forward_from_post_id":fwd_from_post_id,
                                          "channel_title":channel.title, "channel_username":channel.username, "group_type":group_type,
                                          "channel_id": message.peer_id.channel_id, "participants_count": participants_count,
                                          "user_id":user_id, "user_username":user_username, "user_first_name":first_name,
                                          "user_last_name":last_name, "channel_username_chat":channel_username}

                        json.dump(json_tweet, json_file)
                        json_file.write('\n')
                        #except:
                            #continue
                print("Mensagens do canal " + canal + " coletadas.")
                json_file.flush()
