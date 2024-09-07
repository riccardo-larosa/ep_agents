import os
from dotenv import load_dotenv

load_dotenv(override=True)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
#ANTHROPY_API_KEY = os.getenv("ANTHROPY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")