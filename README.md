# Multimodal RAG: OCR-Free Document Retrieval with ColPali & Gemini

An end-to-end Retrieval-Augmented Generation (RAG) system that bypasses traditional OCR. By leveraging the **ColQwen2** Vision-Language Model for Late Interaction (MaxSim) search and **Gemini 2.5 Flash** for generative reasoning, this system can visually "read" and extract answers from complex document layouts, diagrams, and tables.

## 🚀 Key Features
* **OCR-Free Architecture**: Operates directly on document images, preserving spatial layout, charts, and typography that standard text-parsers destroy.
* **Late Interaction Retrieval**: Uses Qdrant vector database to store and compute multi-vector embeddings for highly accurate page-level retrieval.
* **Multimodal Generation**: Integrates Google's Gemini 2.5 Flash API to read the retrieved document images and generate conversational, context-aware answers.
* **Interactive UI**: Built a clean, responsive frontend using Streamlit.

## 📊 Benchmarks
Evaluated on a custom textbook dataset using standard Information Retrieval metrics:
* **MRR (Mean Reciprocal Rank)**: 0.0500 
* **NDCG@5**: 0.0861
*(Note: Zero-shot baseline on a heavily diagram-focused dataset running on CPU architecture).*

## 🛠️ Tech Stack
* **Models**: ColQwen2-v0.1, Google Gemini 2.5 Flash
* **Vector Database**: Qdrant (Docker)
* **Backend Framework**: PyTorch, HuggingFace `colpali-engine`
* **Frontend**: Streamlit
* **Document Processing**: `pdf2image`, Poppler

## 💻 How to Run Locally

1. **Start the Qdrant Database:**
   ```bash
   docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage:z qdrant/qdrant


