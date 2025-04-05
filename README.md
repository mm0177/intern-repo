# Document Analysis Chatbot

This is a Streamlit-based web application that enables users to upload PDF or TXT files and ask questions about their contents. Leveraging a retrieval-augmented generation (RAG) pipeline, the chatbot retrieves relevant document chunks using a hybrid of dense vector search and sparse keyword matching, then generates detailed, structured responses using a large language model.

## Features

- Upload PDF or TXT documents for analysis.
- Ask questions and receive detailed answers based solely on the document context.
- Responses formatted as plain-text bullet points with a concise summary paragraph.
- Powered by advanced NLP models for embedding and text generation.

## Prerequisites

- **Python**: Version 3.8 or higher.
- **GPU (Optional)**: CUDA-enabled GPU (e.g., NVIDIA T4) for faster inference; CPU fallback available.
- **Hugging Face Token**: Required for the gated `meta-llama/Llama-3.2-3B-Instruct` model.
- **Ngrok Token**: For exposing the local Streamlit app publicly.

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
