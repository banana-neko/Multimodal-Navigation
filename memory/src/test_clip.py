import numpy as np
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from pymilvus import MilvusClient, DataType
from PIL import Image
import os
import time


class Memory:
    def __init__(self, collection_name):
        self.client = MilvusClient(
            uri="http://localhost:19530"
        )
        self.collection_name = collection_name
        
        model = ('ViT-H-14-378-quickgelu', 'dfn5b')
        start = time.time()
        self.clip = OpenCLIPEmbeddings(model_name=model[0], checkpoint=model[1])
        print(f"[*] model loaded. ({time.time()-start} sec)")

    def reset(self):
        self.client.drop_collection(
            collection_name=self.collection_name
        )

        schema = MilvusClient.create_schema(
            auto_id=False,
            enabled_dynamic_field=True
        )
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="image_embed", datatype=DataType.FLOAT_VECTOR, dim=1024)
        
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema
        )

        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name="image_embed",
            metric_type="L2",
            index_type="IVF_FLAT",
            index_name="embed_index",
            params={ "nlist": 128 }
        )
        self.client.create_index(
            collection_name=self.collection_name,
            index_params=index_params
        )
    

    def insert(self, ids, image_paths):
        embeds = self.clip.embed_image(image_paths)
        data = [{"id": id, "image_embed": embed} for id, embed in zip(ids, embeds)]

        self.client.insert(
            collection_name=self.collection_name,
            data=data
        )
    

    def search(self, query):
        text_embed = self.clip.embed_documents([query])
        
        self.client.load_collection(
            collection_name=self.collection_name
        )

        res = self.client.search(
            collection_name=self.collection_name,
            anns_field="image_embed",
            data=[text_embed[0]],
            limit=1,
            search_params={"metric_type": "L2"}
        )

        return res


def load_data(dir_path):
    image_files = os.listdir(dir_path)
    image_paths = [os.path.join(dir_path, file_name) for file_name in image_files]
    ids = [int(file_name.split(".")[0]) for file_name in image_files]

    return ids, image_paths


def main():
    memory = Memory("image_collection")
    
    """
    memory.reset()

    image_dir = "/Users/kuzumochi/go2/nav/memory/src/images/test"
    ids, image_paths = load_data(image_dir)

    memory.insert(ids=ids, image_paths=image_paths)
    """

    while True:
        query = input("query> ")
        res = memory.search(query)
        print(res)


if __name__ == "__main__":
    main()