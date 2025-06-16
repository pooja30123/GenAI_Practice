from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()  # Loads from .env

def Tiny_llm():
    llm = ChatOpenAI(
        model="tinyllama-1.1b-chat-v1.0",
        temperature=0.7,
        openai_api_base=os.getenv("OPENAI_API_BASE"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        request_timeout=300,  # 5 minutes timeout
        max_retries=2
    )
    return llm

def Thebloke_llm():
    llm = ChatOpenAI(
        model="thebloke/mistral-7b-instruct-v0.1",  # Updated to match LM Studio logs
        temperature=0.7,
        openai_api_base=os.getenv("OPENAI_API_BASE"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        request_timeout=300,  # 5 minutes timeout
        max_retries=2
    )
    return llm

def Mistral_llm():
    llm = ChatOpenAI(
        model="thebloke/mistral-7b-instruct-v0.1",  # Updated to match LM Studio logs
        temperature=0.7,
        openai_api_base=os.getenv("OPENAI_API_BASE"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        request_timeout=300,  # 5 minutes timeout
        max_retries=2
    )
    return llm

def nomicEmbedding():
    llm = ChatOpenAI(
        model="text-embedding-nomic-embed-text-v1.5",  
        temperature=0.7,
        openai_api_base=os.getenv("OPENAI_API_BASE"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        request_timeout=300,  
    )
    return llm
