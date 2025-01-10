from pymilvus import MilvusClient

client = MilvusClient(uri="http://localhost:19530")
client.drop_collection(
    collection_name="image_collection"
)