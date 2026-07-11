from langchain.chat_models import init_chat_model
from dotenv import load_dotenv, find_dotenv
from os import getenv


load_dotenv(find_dotenv())
model = init_chat_model(
    api_key=getenv("DEEPSEEK_API_KEY"),
    base_url=getenv("DEEPSEEK_API_BASE_URL"),
    model="deepseek-chat",
    temperature=0,
    timeout=30,
    max_retries=2,
)
