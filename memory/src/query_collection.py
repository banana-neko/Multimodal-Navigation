from pymilvus import MilvusClient
from langchain_experimental.open_clip import OpenCLIPEmbeddings


client = MilvusClient(
    uri="http://localhost:19530",
)

# 4. Single vector search
clip = OpenCLIPEmbeddings(model_name="ViT-B-32", checkpoint="laion2b_s34b_b79k")
query = "玄関扉が見える場所"
text_embed = clip.embed_documents([query])

res = client.search(
    collection_name="image_collection",
    anns_field="image_embed",
    data=text_embed,
    limit=1,
    search_params={"metric_type": "L2"}
)

for hits in res:
    for hit in hits:
        print(hit)