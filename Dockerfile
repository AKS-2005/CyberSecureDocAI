# Use a stable, slim version of Python
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install cmake and build tools
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake

# Copy the requirements file first
COPY requirements.txt .

# Build llama-cpp-python with CPU hardware acceleration
RUN CMAKE_ARGS="-DLLAMA_AVX2=ON -DLLAMA_FMA=ON" pip install --no-cache-dir --default-timeout=600 -r requirements.txt

# --- PRE-DOWNLOAD ALL AI MODELS FOR FULL OFFLINE USE ---
# 1. Download the main LLM (Qwen2)
RUN mkdir -p /app/models
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('Qwen/Qwen2-1.5B-Instruct-GGUF', 'qwen2-1_5b-instruct-q4_k_m.gguf', local_dir='/app/models', local_dir_use_symlinks=False)"

# 2. Download the embedding model
RUN python3 -c "from langchain_huggingface import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')"

# Copy the rest of the application code
COPY . .

# Create the persistent data directory
RUN mkdir -p /app/data

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]
