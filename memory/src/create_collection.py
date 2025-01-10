from pymilvus import MilvusClient, DataType

client = MilvusClient(
    uri="http://localhost:19530"
)

schema = MilvusClient.create_schema(
    auto_id=False,
    enabled_dynamic_field=True
)

"""
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="image_embedding", datatype=DataType.FLOAT_VECTOR, dim=512)
schema.add_field(field_name="position", datatype=DataType.FLOAT_VECTOR, dim=3)
"""

schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="image_embedding", datatype=DataType.FLOAT_VECTOR, dim=512)

client.create_collection(
    collection_name="image_collection",
    schema=schema
)