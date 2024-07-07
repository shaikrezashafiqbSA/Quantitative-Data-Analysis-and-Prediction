#settings.py
from dotenv import load_dotenv
load_dotenv()
import os 
import os

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
METAAPI_API_KEY = os.getenv("METAAPI_API_KEY")
METAAPI_ID = os.getenv("METAAPI_ID")

WEBHOOK_URL = os.getenv("WEBHOOK_URL")
KGI_URL = os.getenv("KGI_URL")
KGI_BEARER = os.getenv("KGI_BEARER")



DB_USER = os.getenv("DB_USER")
DB_NAME = os.getenv("DB_NAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_SCHEMA = os.getenv("DB_SCHEMA")

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# BARD_API_KEY = os.getenv("BARD_API_KEY")
# BARD_API_KEY2 = os.getenv("BARD_API_KEY2")

# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
# PINECONE_INDEX = os.getenv("PINECONE_INDEX")
# PINECONE_DIMENSION = int(os.getenv("PINECONE_DIMENSION"))

# S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
# S3_SECRET_ACCESS_KEY = os.getenv("S3_SECRET_ACCESS_KEY")
# S3_ACCESS_KEY_ID = os.getenv("S3_ACCESS_KEY_ID")
# S3_REGION = os.getenv("S3_REGION")