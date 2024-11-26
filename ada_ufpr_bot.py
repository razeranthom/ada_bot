#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os
import re
import shutil
from os.path import join, dirname
from string import Template
from typing import List
from emoji import emojize
from dotenv import load_dotenv

# Biblioteca da OpenAI
import openai

# Biblioteca do ChromaDB
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.utils import embedding_functions
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
import chromadb

# Biblioteca do Telegram 
from telegram.constants import ParseMode, ChatAction
from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove, Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes


# Dados do Chroma/RAG

# Se é para usar RAG ou contexto no prompt
USE_RAG = False 
# Se é para recriar o BD toda inicialização do script
RESET_CHROMA = True

CHROMA_PATH = "chromadb" 
DB_NAME = "ada_db"
DB_PATH = os.path.join(os.getcwd(), CHROMA_PATH)
EMBEDDING_FUNCTION = embedding_functions.DefaultEmbeddingFunction()

# Dados de arquivos
FILE_PREFIX = "txt_"
CONTEXTO_PATH = "./dados_curso.txt"
PROMPT_PATH = "./prompt_template.txt"


#######################################################################
# Funções Auxiliares 

# Carrega um arquivo e retorna
def load_file(nome):
    with open(nome, 'r') as myfile:
        data = myfile.read()
    return data

# Carrega um arquivo com prefixo e retorna
def load_file_prefix(nome):
    return load_file(FILE_PREFIX + nome + '.txt')

# Função auxiliar para obter informação do usuário
def get_user_chat_info(update, context):
    return update.effective_user.id, update.effective_chat.type

# Função de log 
def log(acao, update, context):
    usr_name, first_name, usr_id = get_user_name(update, context)
    logger.info('Usuário: @%s (%s) - %s; Acesso: "%s"', usr_name, str(usr_id), first_name, acao)


#######################################################################
# Inicialização

# Carrega:
# MARITALK_API_KEY, MARITALK_MODEL, MARITALK_URL
# BOT_TOKEN, BOT_NAME, BOT_LINK
load_dotenv(join(dirname(__file__), '.env'))

MARITALK_API_KEY = os.environ.get("MARITALK_API_KEY")
MARITALK_MODEL = os.environ.get("MARITALK_MODEL")
MARITALK_URL = os.environ.get("MARITALK_URL")
BOT_TOKEN = os.environ.get("BOT_TOKEN")
BOT_NAME = os.environ.get("BOT_NAME")
BOT_LINK = os.environ.get("BOT_LINK")

# Carrega o contexto: dados do curso
ARQ_CONTEXTO = load_file(CONTEXTO_PATH)
# Carrega o template de prompt 
PROMPT_TEMPLATE = load_file(PROMPT_PATH)

# Cria o log 
logging.basicConfig(filename='ada_ufpr_bot.log', format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

#######################################################################
# Funções que implementam o RAG 

def word_splitter(source_text: str) -> List[str]:
    source_text = re.sub("\s+", " ", source_text) 
    return re.split("\s", source_text)

def get_chunks_overlap(text: str, chunk_size: int, overlap_fraction: float) -> List[str]:
    text_words = word_splitter(text)
    overlap_int = int(chunk_size * overlap_fraction)
    chunks = []
    for i in range(0, len(text_words), chunk_size):
        chunk_words = text_words[max(i - overlap_int, 0): min(i + chunk_size + overlap_int, len(text_words))]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)
    return chunks

# Separa um texto em partes
# Pode-se usar várias estratégias
def split_text(text):
    return get_chunks_overlap(text, 300, 0.3)
    # return [i for i in re.split('\n\n', text) if i.strip()]

# Cria o banco de dados com as porções de documentos
def create_chroma_db(documents: List[str], path: str, name: str):
    chroma_client = chromadb.PersistentClient(path=path)
    db = chroma_client.create_collection(name=name, embedding_function=EMBEDDING_FUNCTION)
    for i, d in enumerate(documents):
        db.add(documents=[d], ids=[str(i)])
    return db

# Carrega o banco 
def load_chroma_collection(path: str, name: str):
    chroma_client = chromadb.PersistentClient(path=path)
    return chroma_client.get_collection(name=name, embedding_function=EMBEDDING_FUNCTION)

# Recupera os conteúdos mais relevantes do banco 
def get_relevant_text(query: str, db, n_results: int):
    results = db.query(query_texts=[query], n_results=n_results)
    return [doc[0] for doc in results['documents']]

# Inicializa o RAG
def initialize_rag():
    # Cria o diretório para o chroma 
    if not os.path.exists(CHROMA_PATH):
        os.makedirs(CHROMA_PATH)
    else:
        if RESET_CHROMA:
            shutil.rmtree(CHROMA_PATH)
            os.makedirs(CHROMA_PATH)

    chunked_text = split_text(ARQ_CONTEXTO)
    db = create_chroma_db(chunked_text, DB_PATH, DB_NAME)
 
#######################################################################
# Funções de chamada do LLM 

# Chama a API da Maritaca AI
# O correto é ajustar as mensagens para as roles corretas
def query_maritalk(query_text):
   client = openai.OpenAI( api_key=MARITALK_API_KEY,
                           base_url=MARITALK_URL)
   messages = [
       {"role": "user", "content": query_text},
   ]
   response = client.chat.completions.create(
         model=MARITALK_MODEL,
         messages=messages,
         temperature=0.7,
         max_tokens=1700,
   )
   answer = response.choices[0].message.content
   return answer

# Responde uma mensagem usando RAG 
def query_rag(query_text):
  # Carrega o banco
  db = load_chroma_collection(DB_PATH, DB_NAME)
  # Busca n textos relevantes
  relevant_text = get_relevant_text(query_text, db, n_results=3)
  if not relevant_text:
      rag_context = "Não foram encontradas informações sobre este assunto." 
  else:
      rag_context = "".join(relevant_text)
  # Monta o prompt para o LLM
  template = Template(PROMPT_TEMPLATE)
  prompt = template.substitute(context=rag_context, question=query_text)
  response_text = query_maritalk(prompt)
  return response_text

# Responde uma mensagem usando o contexto no prompt 
def query_context(query_text):
  # Cria o prompt usando o contexto e a consulta 
  template = Template(PROMPT_TEMPLATE)
  prompt = template.substitute(context=ARQ_CONTEXTO, question=query_text)
  response_text = query_maritalk(prompt)
  return response_text

# Responde uma mensagem recebida no Telegram
# Aqui podemos chavear as estratégias
def query_message(txt):
    #txt = guardrail(txt)
    if USE_RAG:
        resp = query_rag(txt) 
    else:
        resp = query_context(txt)
    #resp = guardrail(resp)
    return resp

   
#######################################################################
# Implementação do BOT

# Trata o comando /start 
async def bot_start(update, context):
    """Send a message when the command /start is issued."""
    msg = load_file_prefix("start")
    await context.bot.send_message(chat_id=update.effective_chat.id, text=msg, parse_mode=ParseMode.MARKDOWN, disable_web_page_preview=True, reply_markup=ReplyKeyboardRemove() )

# Trata o comando /help
async def bot_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Send the message with menu
    msg = "Converse comigo sobre o Curso de Especialização em IA Aplicada!!!"
    await update.message.reply_markdown(msg)
        
# Trata a mensagem recebida pelo Telegram e pede uma resposta
async def bot_message(update, context):
    """Answer User message."""
    # Mensagem que o usuário digitou 
    msg = update.message.text
    # Dados do usuário, se quiser fazer alguma verificação 
    usr_id, chat_type = get_user_chat_info(update, context)
    # Se for uma mensagem que cita o Bot, retira nome dele da pergunta
    msg = msg.replace("@"+BOT_NAME, "")
    # Manda uma ação "Digitando..."
    await context.bot.send_chat_action(chat_id=usr_id, action=ChatAction.TYPING)
    # Solicita uma resposta
    msg_r = query_message(msg)
    # Manda a resposta para o Telegram
    await context.bot.send_message(chat_id=usr_id, text=msg_r, parse_mode=ParseMode.MARKDOWN) 

# Tratamento de erros, só loga por enquanto
def bot_error(update, context):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, context.error)

# Inicializa o BOT
def main():
    """Start the bot."""
    if USE_RAG:
        initialize_rag()

    # Inicializa o BOT
    application = Application.builder().token(BOT_TOKEN).build()

    # Instala handlers para os vários comandos 
    application.add_handler(CommandHandler("start", bot_start))
    application.add_handler(CommandHandler("help", bot_help))

    # Se receber uma mensagem, chama a função correta
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot_message))

    # Roda o BOT
    application.add_error_handler(bot_error)
    application.run_polling(allowed_updates=Update.ALL_TYPES) 

if __name__ == '__main__':
    main()
