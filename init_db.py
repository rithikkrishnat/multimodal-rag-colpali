from qdrant_client import QdrantClient
from qdrant_client.http import models

def initialize_qdrant_collection():
    # 1. Connect to the local Qdrant instance
    client = QdrantClient(url="http://localhost:6333")
    collection_name = "financial_documents"

    # 2. Check if the collection already exists to prevent overwriting data
    if client.collection_exists(collection_name=collection_name):
        print(f"Collection '{collection_name}' already exists. Ready for insertion.")
        return client, collection_name

    print(f"Creating new collection: '{collection_name}'...")

    # 3. Create the collection with ColPali-specific architecture
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=128,  # ColPali-v1.3 outputs 128-dimensional vectors
            distance=models.Distance.COSINE, # Cosine similarity is standard for these embeddings
            
            # THE MOST IMPORTANT PART: Enabling Late Interaction (MaxSim)
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM
            )
        )
    )
    
    print("Database successfully initialized for Multi-Vector Late Interaction!")
    return client, collection_name

if __name__ == "__main__":
    # Run the initialization
    db_client, name = initialize_qdrant_collection()
    
    # Verify the setup
    collection_info = db_client.get_collection(collection_name=name)
    print(f"Collection Status: {collection_info.status}")