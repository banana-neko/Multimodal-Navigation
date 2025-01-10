from langchain_openai import OpenAIEmbeddings
from langchain_milvus import Milvus
from langchain_core.documents import Document
from uuid import uuid4

document_1 = Document(
    page_content="扉と傘立てがある。",
    metadata={"position": "[0, 1, 0, 15]"}
)

document_2 = Document(
    page_content="ゴミ箱がある。",
    metadata={"position": "[0.06, 1.89, 0, 12]"}
)

document_3 = Document(
    page_content="黒いソファとその横に観葉植物が置いてある。",
    metadata={"position": "[0, 0, 0, 30]"}
)

documents = [
    document_1,
    document_2,
    document_3,
]

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = Milvus.from_documents(
    documents,
    embeddings,
    connection_args={"uri": "http://localhost:19530"},
    collection_name="test_db"
)

uuids = [str(uuid4()) for _ in range(len(documents))]
vector_store.add_documents(documents=documents, ids=uuids)

results = vector_store.similarity_search(
    "座れる場所",
    k=1
)

for res in results:
    print(f"* {res.page_content} [{res.metadata}]")