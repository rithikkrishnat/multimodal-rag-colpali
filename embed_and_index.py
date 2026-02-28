import os
import torch
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.http import models

# 1. Import from the official colpali_engine instead of transformers
from colpali_engine.models import ColQwen2, ColQwen2Processor

# 2. Use the open-source, higher-performing ColQwen2 model (No Hugging Face token required!)
MODEL_NAME = "vidore/colqwen2-v0.1" 
IMAGE_DIR = "processed_images"
COLLECTION_NAME = "financial_documents"
BATCH_SIZE = 1 # Process 1 page at a time to protect your CPU's RAM

def load_model():
    print("Loading AI Model... (Downloading ~5GB, this only happens once!)")
    
    # Safely fallback to CPU since you don't have a dedicated Nvidia GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32 
    print(f"Using device: {device.upper()}")
    
    model = ColQwen2.from_pretrained(
        MODEL_NAME, 
        torch_dtype=dtype, 
        device_map=device
    ).eval()
    
    processor = ColQwen2Processor.from_pretrained(MODEL_NAME)
    return model, processor, device

def embed_and_store():
    client = QdrantClient(url="http://localhost:6333")
    model, processor, device = load_model()
    
    if not os.path.exists(IMAGE_DIR):
        print(f"Error: Folder '{IMAGE_DIR}' not found.")
        return
        
    image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")])
    if not image_files:
        print("No images found to process.")
        return

    print(f"Found {len(image_files)} pages. Starting embedding process...")

    for i in range(0, len(image_files), BATCH_SIZE):
        batch_files = image_files[i : i + BATCH_SIZE]
        print(f"Processing page(s) {i+1} to {min(i+BATCH_SIZE, len(image_files))}...")
        
        images = [Image.open(os.path.join(IMAGE_DIR, f)) for f in batch_files]
        
        # 3. Use the official process_images method
        inputs = processor.process_images(images).to(device)
        
        with torch.no_grad():
            embeddings = model(**inputs)
            
        points = []
        for j, embedding in enumerate(embeddings):
            page_index = i + j + 1
            # Convert to standard Python float list for database storage
            vector_data = embedding.cpu().float().numpy().tolist()
            
            points.append(
                models.PointStruct(
                    id=page_index,
                    vector=vector_data,
                    payload={
                        "page_number": page_index,
                        "filename": batch_files[j]
                    }
                )
            )
            
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f" Successfully saved page to Qdrant database.")

    print("\n All pages embedded and indexed successfully!")

if __name__ == "__main__":
    embed_and_store()