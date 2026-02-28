import torch
from qdrant_client import QdrantClient
from colpali_engine.models import ColQwen2, ColQwen2Processor

# Configuration
MODEL_NAME = "vidore/colqwen2-v0.1"
COLLECTION_NAME = "financial_documents"

def load_model():
    print("Loading AI Model for Search...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32 
    
    # Loads from your local cache, so no 5GB download this time!
    model = ColQwen2.from_pretrained(
        MODEL_NAME, 
        torch_dtype=dtype, 
        device_map=device
    ).eval()
    
    processor = ColQwen2Processor.from_pretrained(MODEL_NAME)
    return model, processor, device

def search_database(query_text):
    client = QdrantClient(url="http://localhost:6333")
    model, processor, device = load_model()
    
    print(f"\nSearching for: '{query_text}'")
    print("Embedding your question...")
    
    # 1. Process the text query
    inputs = processor.process_queries([query_text]).to(device)
    
    # 2. Generate the vector for the question
    with torch.no_grad():
        query_embeddings = model(**inputs)
        
    # ColQwen returns a tensor, we need the first one (since we only asked 1 question)
    query_vector = query_embeddings[0].cpu().float().numpy().tolist()
    
    print("Scanning document pages using Late Interaction (MaxSim)...")
    
    # 3. Search Qdrant
    search_results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=3 # Return the top 3 best matching pages
    )
    
    # 4. Display the results
    print("\n" + "="*50)
    print("SEARCH RESULTS")
    print("="*50)
    
    if not search_results.points:
        print("No matches found. Is your database empty?")
        return

    for rank, result in enumerate(search_results.points, 1):
        page_num = result.payload.get("page_number", "Unknown")
        filename = result.payload.get("filename", "Unknown")
        score = result.score
        
        print(f"Rank {rank}:")
        print(f"Document Page: {page_num}")
        print(f"Image File: {filename}")
        print(f"Match Score: {score:.4f}")
        print("-" * 30)

if __name__ == "__main__":
    my_question = "explain structure of a national isp"
    
    search_database(my_question)


