from langchain_openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

llm = OpenAI(model='gpt-3.5-turbo-instruct')
reuslt = llm.invoke("What is the capital of India?")
print(reuslt)