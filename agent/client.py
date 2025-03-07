import os

from openai import OpenAI

from dotenv import load_dotenv

load_dotenv('GPT_agent.env')

client = OpenAI(
    api_key = os.getenv('OPENAI_CLIENT_KEY'),
    base_url = "https://api.fe8.cn/v1"
)