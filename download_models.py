"""
CyberSecureDocAI - Offline Model Downloader
Downloads Qwen2-1.5B-Instruct-GGUF and sentence-transformers/all-MiniLM-L6-v2
to prepare the system for completely offline, air-gapped operation.
"""

import os
import sys

def download_models():
    print("=" * 60)
    print(" CyberSecureDocAI - Offline Model Setup")
    print("=" * 60)
    
    # 1. Setup Models directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    llm_filename = "qwen2-1_5b-instruct-q4_k_m.gguf"
    llm_filepath = os.path.join(models_dir, llm_filename)
    
    print(f"\n[1/2] Checking Local LLM: {llm_filename}")
    if os.path.exists(llm_filepath) and os.path.getsize(llm_filepath) > 500_000_000:
        print(f"       LLM model already present: {llm_filepath} ({os.path.getsize(llm_filepath)/(1024*1024):.1f} MB)")
    else:
        print("       Downloading Qwen2-1.5B-Instruct GGUF from Hugging Face...")
        print("       Repository: Qwen/Qwen2-1.5B-Instruct-GGUF")
        try:
            from huggingface_hub import hf_hub_download
            downloaded_path = hf_hub_download(
                repo_id="Qwen/Qwen2-1.5B-Instruct-GGUF",
                filename=llm_filename,
                local_dir=models_dir,
                local_dir_use_symlinks=False
            )
            print(f"       Download complete: {downloaded_path}")
        except Exception as e:
            print(f"       ERROR downloading LLM model: {e}")
            print("       You can manually place 'qwen2-1_5b-instruct-q4_k_m.gguf' in the 'models/' folder.")
            return False

    print("\n[2/2] Checking Embedding Model: sentence-transformers/all-MiniLM-L6-v2")
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        print("       Initializing and caching embedding model locally...")
        _ = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        print("       Embedding model successfully cached and ready for offline use.")
    except Exception as e:
        print(f"       Warning while caching embeddings: {e}")
        try:
            from sentence_transformers import SentenceTransformer
            _ = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            print("       Fallback sentence-transformers cache completed successfully.")
        except Exception as err:
            print(f"       ERROR downloading embeddings: {err}")
            return False

    print("\n" + "=" * 60)
    print(" ALL MODELS ARE READY FOR SECURE OFFLINE INFERENCE!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = download_models()
    if not success:
        sys.exit(1)
