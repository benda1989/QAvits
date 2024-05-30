import openai
from redis import Redis
from llama_index.vector_stores.redis import RedisVectorStore
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, SimpleDirectoryReader
from llama_index.readers.file import DocxReader
import logging
import sys


# init doc vector
# vector_store = RedisVectorStore(redis_client=Redis.from_url("redis://localhost:6379"), overwrite=True)
# documents = SimpleDirectoryReader("./data", file_extractor={".docx": DocxReader()}).load_data()
# index = VectorStoreIndex.from_documents(documents, storage_context=StorageContext.from_defaults(vector_store=vector_store))
# exit()

# vector from redis
vector_store = RedisVectorStore(redis_client=Redis.from_url("redis://localhost:6379"))  # set your rag data
index = VectorStoreIndex.from_vector_store(vector_store)
query_engine = index.as_query_engine()
