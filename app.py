import streamlit as st
import torch
import os
from PIL import Image
from qdrant_client import QdrantClient
from colpali_engine.models import ColQwen2, ColQwen2Processor

# 1. Import the NEW official Google SDK
from google import genai 

# --- Configuration ---
MODEL_NAME = "vidore/colqwen2-v0.1"
COLLECTION_NAME = "financial_documents"
IMAGE_DIR = "processed_images"

# ⚠️ PASTE YOUR GEMINI API KEY HERE ⚠️
GEMINI_API_KEY = "AIzaSyB2fHcpWeTTZASAK4iXkOe5M7Lb0mtkeXY"

# 2. Initialize the new client
ai_client = genai.Client(api_key=GEMINI_API_KEY)

# --- Setup Web Page ---
st.set_page_config(page_title="ColPali + Gemini RAG", layout="wide")
st.title("🤖 Multimodal RAG: ColPali & Gemini")
st.markdown("ColPali retrieves the exact visual page, and Gemini reads the image to answer your question.")

# --- Load ColPali Model Once ---
@st.cache_resource(show_spinner=False)
def load_ai_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32 
    
    model = ColQwen2.from_pretrained(MODEL_NAME, torch_dtype=dtype, device_map=device).eval()
    processor = ColQwen2Processor.from_pretrained(MODEL_NAME)
    client = QdrantClient(url="http://localhost:6333")
    
    return model, processor, client, device

with st.spinner("Loading Vision Database..."):
    model, processor, client, device = load_ai_model()

# --- User Interface ---
st.divider()
query_text = st.text_input("🔍 Ask a question about your document:", placeholder="e.g., Explain what ARPANET is and why it was created.")

if st.button("Generate Answer", type="primary"):
    if query_text.strip():
        
        # --- PHASE 1: RETRIEVAL (ColQwen2) ---
        with st.spinner("ColPali is scanning the document layout..."):
            inputs = processor.process_queries([query_text]).to(device)
            with torch.no_grad():
                query_embeddings = model(**inputs)
            query_vector = query_embeddings[0].cpu().float().numpy().tolist()

            search_results = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=1 # We only need the #1 best matching page for the AI to read
            )
        
        # --- PHASE 2: GENERATION (Gemini 2.5 Flash) ---
        if search_results.points:
            best_match = search_results.points[0]
            filename = best_match.payload.get("filename")
            page_num = best_match.payload.get("page_number")
            img_path = os.path.join(IMAGE_DIR, filename)
            
            if os.path.exists(img_path):
                retrieved_image = Image.open(img_path)
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("💡 AI Answer")
                    with st.spinner("Gemini is reading the retrieved page..."):
                        try:
                            prompt = f"You are a helpful assistant. Look at the provided document page and answer the user's question. If the answer is not in the image, say 'I cannot find the answer on this page.'\n\nQuestion: {query_text}"
                            
                            # 3. Use the new generate_content syntax
                            response = ai_client.models.generate_content(
                                model='gemini-2.5-flash',
                                contents=[prompt, retrieved_image]
                            )
                            
                            st.success("Response Generated!")
                            st.write(response.text)
                        except Exception as e:
                            st.error(f"Gemini API Error: {e}")
                
                with col2:
                    st.subheader(f"📄 Source Evidence (Page {page_num})")
                    st.image(retrieved_image, use_container_width=True)
            else:
                st.error("Error: Retrieved image file not found on disk.")
        else:
            st.warning("No relevant pages found in the database.")