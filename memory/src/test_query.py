from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store_loaded = Milvus(
    embeddings,
    connection_args={"uri": "http://localhost:19530"},
    collection_name="test_db",
)