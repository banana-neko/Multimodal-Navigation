from pymilvus import MilvusClient

client = MilvusClient(uri="http://localhost:19530")

collections = client.list_collections()
for collection in collections:
    print(collection)