import torch
import math
import time
from qdrant_client import QdrantClient
from colpali_engine.models import ColQwen2, ColQwen2Processor

# --- Configuration ---
MODEL_NAME = "vidore/colqwen2-v0.1"
COLLECTION_NAME = "financial_documents"

# --- 1. The Golden Dataset (Ground Truth) ---
# I mapped these directly to the actual pages in your sample_report.pdf!
GROUND_TRUTH = {
    "Show me the diagram of a mesh topology.": [6],
    "What are the differences between OSI and TCP/IP Reference Models?": [24],
    "Explain what ARPANET is and why it was created.": [25],
    "How does a microwave transmission work with repeaters?": [29],
    "Compare connection-oriented and connection-less services.": [19]
}

def load_ai_model():
    print("Loading AI Model for Benchmarking...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32 
    
    model = ColQwen2.from_pretrained(MODEL_NAME, torch_dtype=dtype, device_map=device).eval()
    processor = ColQwen2Processor.from_pretrained(MODEL_NAME)
    client = QdrantClient(url="http://localhost:6333")
    
    return model, processor, client, device

# --- 2. Metric Calculation Functions ---
def calculate_rr(retrieved_pages, correct_pages):
    """Calculates Reciprocal Rank for a single query."""
    for rank, page in enumerate(retrieved_pages, start=1):
        if page in correct_pages:
            return 1.0 / rank
    return 0.0

def calculate_ndcg(retrieved_pages, correct_pages, k=5):
    """Calculates NDCG@k for a single query."""
    dcg = 0.0
    for rank, page in enumerate(retrieved_pages[:k], start=1):
        if page in correct_pages:
            # Relevance is 1 if it's the correct page, 0 otherwise
            dcg += 1.0 / math.log2(rank + 1)
            
    # IDCG (Ideal DCG) is simply 1.0 since we only have 1 "perfect" relevant page per query
    idcg = 1.0 
    return dcg / idcg

# --- 3. The Benchmarking Pipeline ---
def run_benchmark():
    model, processor, client, device = load_ai_model()
    
    total_mrr = 0.0
    total_ndcg = 0.0
    total_latency = 0.0
    num_queries = len(GROUND_TRUTH)
    
    print("\n" + "="*50)
    print("🚀 STARTING COLPALI BENCHMARK")
    print("="*50)

    for query, correct_pages in GROUND_TRUTH.items():
        print(f"\nQuery: '{query}'")
        print(f"Expected Page(s): {correct_pages}")
        
        # Start timer for latency tracking
        start_time = time.time()
        
        # Embed Query
        inputs = processor.process_queries([query]).to(device)
        with torch.no_grad():
            query_embeddings = model(**inputs)
        query_vector = query_embeddings[0].cpu().float().numpy().tolist()

        # Search Qdrant
        search_results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=5 # Get Top 5 for NDCG@5
        )
        
        # End timer
        latency = time.time() - start_time
        total_latency += latency
        
        # Extract the retrieved page numbers
        retrieved_pages = [hit.payload.get("page_number") for hit in search_results.points]
        print(f"Retrieved Pages: {retrieved_pages} (Time: {latency:.2f}s)")
        
        # Calculate Metrics
        rr = calculate_rr(retrieved_pages, correct_pages)
        ndcg = calculate_ndcg(retrieved_pages, correct_pages, k=5)
        
        total_mrr += rr
        total_ndcg += ndcg
        
        print(f"Metric -> RR: {rr:.3f} | NDCG@5: {ndcg:.3f}")

    # --- 4. Final Research Report Output ---
    avg_mrr = total_mrr / num_queries
    avg_ndcg = total_ndcg / num_queries
    avg_latency = total_latency / num_queries
    
    print("\n" + "="*50)
    print("📊 FINAL RESEARCH METRICS")
    print("="*50)
    print(f"Total Queries Evaluated: {num_queries}")
    print(f"Mean Reciprocal Rank (MRR):  {avg_mrr:.4f}  <-- (Closer to 1.0 is better)")
    print(f"Average NDCG@5:              {avg_ndcg:.4f}  <-- (Closer to 1.0 is better)")
    print(f"Average Inference Latency:   {avg_latency:.2f} seconds/query")
    print("="*50)

if __name__ == "__main__":
    run_benchmark()