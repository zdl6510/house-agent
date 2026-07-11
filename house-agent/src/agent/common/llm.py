from langchain.chat_models import init_chat_model
from dotenv import load_dotenv, find_dotenv
from os import getenv


load_dotenv(find_dotenv())
model = init_chat_model(
    api_key=getenv("DEEPSEEK_API_KEY"),
    base_url=getenv("DEEPSEEK_API_BASE_URL"),
    model=getenv("DEEPSEEK_MODEL", "deepseek-chat"),
    temperature=0,
    timeout=float(getenv("LLM_TIMEOUT_SECONDS", "30")),
    max_retries=int(getenv("LLM_MAX_RETRIES", "2")),
)
