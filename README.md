# 🛡️ CyberSecureDocAI: Secure Offline RAG Document Analysis System

**A Capstone Project — School of Computing Science and Artificial Engineering, VIT Bhopal University**

### 👥 Project Authors
- Anuj Nagar (23BCE10244)
- Akshiv (23BCE10785)
- Sameer Kumar (23BCE11610)
- Chirayu Zalke (23BCE11752)
- Tarun (23BCE10932)
- **Project Supervisor:** Dr. G.R. Hemalakshmi

---

## 📌 Abstract & Overview
**CyberSecureDocAI** is a private, multi-user, 100% offline Retrieval-Augmented Generation (RAG) chatbot designed for sensitive document analysis (e.g., cybersecurity threat intelligence, forensic reports, compliance guidelines, and confidential enterprise documentation).

By eliminating external cloud APIs, the system guarantees zero data exfiltration, strict data isolation between accounts, and resilient air-gapped performance on standard CPU hardware.

---

## 🏗️ System Architecture

```
                 [ User Web Browser ]
                         │
             HTTP Requests / REST API
                         │
        ┌────────────────▼────────────────┐
        │       Flask Web Application     │
        │   (Authentication & Controllers)│
        └───────┬────────────────┬────────┘
                │                │
    ┌───────────▼────────┐   ┌───▼──────────────────┐
    │ SQLite User DB     │   │ Multi-format Ingest  │
    │ (SHA-256 Auth &    │   │ (PDF / TXT / MD)     │
    │ Session Control)   │   └───────────┬──────────┘
    └────────────────────┘               │
                             ┌───────────▼──────────┐
                             │ Recursive Chunking   │
                             │ (800 size, 150 lap)  │
                             └───────────┬──────────┘
                                         │
                             ┌───────────▼──────────┐
                             │ SentenceTransformer  │
                             │ (all-MiniLM-L6-v2)   │
                             └───────────┬──────────┘
                                         │
                             ┌───────────▼──────────┐
                             │ FAISS Vector Store   │
                             │ (Isolated User Index)│
                             └───────────┬──────────┘
                                         │ Similarity Retrieval (Top-3)
                             ┌───────────▼──────────┐
                             │ Qwen2-1.5B Instruct  │
                             │ (LlamaCpp AVX2 CPU)  │
                             └───────────┬──────────┘
                                         │
                             ┌───────────▼──────────┐
                             │ Grounded Response +  │
                             │ Document Citations   │
                             └──────────────────────┘
```

---

## ⚙️ Requirements

### Hardware Requirements
- **CPU:** Multi-core Intel Core i5/i7 or AMD Ryzen 5/7 with **AVX2 / FMA** support.
- **RAM:** Minimum **16 GB RAM**.
- **Storage:** 20–50 GB free disk space.
- **GPU:** Optional (CPU inference is fully optimized with quantized GGUF).

### Software Requirements
- **Python:** 3.10+ / 3.11
- **Docker:** Docker Desktop 20.10+ (if deploying via container)
- **C++ Build Tools & CMake:** (only if compiling `llama-cpp-python` locally on Windows from source)

---

## 🚀 Quick Start Guide

### Option A: 1-Click Launch with Docker (Recommended)
This method containerizes the Python app, downloads the models in the build step, and runs on `http://localhost:5000`.

```bash
docker compose up --build
```
*(Or double-click `run_docker.bat` on Windows)*

---

### Option B: Local Native Windows / Python Run
If you prefer running directly on your host machine without Docker:

1. **Install Dependencies:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Download Offline Models:**
   ```bash
   python download_models.py
   ```
   *Downloads `qwen2-1_5b-instruct-q4_k_m.gguf` (~980MB) and pre-caches `all-MiniLM-L6-v2`.*

3. **Launch the Server:**
   ```bash
   python app.py
   ```
   *Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your web browser.*

---

## 🔒 Security & Isolation Features
- **User Segregation:** Uploads and vector databases are siloed into `/data/uploads/user_<id>` and `/data/vectorstores/user_<id>`.
- **Zero Cloud Leakage:** All LLM queries and document embeddings remain on localhost.
- **Verified Citations:** Every AI answer provides citations linking back to the exact chunk text extracted from the user's documents.
- **Audit History:** Past interactions are saved per-user with full clear-history control.
