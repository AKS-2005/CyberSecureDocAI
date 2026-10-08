import os
import shutil
import traceback
import sqlite3
import hashlib
import time
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory, session
from werkzeug.utils import secure_filename
from pypdf import PdfReader

# LangChain Imports with backward & forward compatibility
try:
    from langchain_core.prompts import PromptTemplate
except ImportError:
    try:
        from langchain.prompts import PromptTemplate
    except ImportError:
        from langchain.prompts.prompt import PromptTemplate

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
from langchain_huggingface import HuggingFaceEmbeddings

try:
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    try:
        from langchain.chains import create_stuff_documents_chain
    except ImportError:
        class SimpleStuffChain:
            def __init__(self, llm, prompt):
                self.llm = llm
                self.prompt = prompt
            def invoke(self, inputs):
                docs = inputs.get("context", [])
                user_input = inputs.get("input", "")
                context_str = "\n\n".join([d.page_content if hasattr(d, 'page_content') else str(d) for d in docs])
                formatted_prompt = self.prompt.format(context=context_str, input=user_input)
                if hasattr(self.llm, 'invoke'):
                    res = self.llm.invoke(formatted_prompt)
                else:
                    res = self.llm(formatted_prompt)
                return res

        def create_stuff_documents_chain(llm, prompt):
            return SimpleStuffChain(llm, prompt)

# --- BASE & PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get('DATA_DIR', os.path.join(BASE_DIR, 'data'))
UPLOAD_FOLDER_BASE = os.path.join(DATA_DIR, 'uploads')
VECTORSTORE_BASE = os.path.join(DATA_DIR, 'vectorstores')
DATABASE_PATH = os.path.join(DATA_DIR, 'users.db')

# Model path resolution: check ENV, then local models folder, then docker path
DEFAULT_MODEL_FILENAME = "qwen2-1_5b-instruct-q4_k_m.gguf"
ENV_MODEL_PATH = os.environ.get('MODEL_PATH')
LOCAL_MODEL_PATH = os.path.join(BASE_DIR, 'models', DEFAULT_MODEL_FILENAME)
DOCKER_MODEL_PATH = f"/app/models/{DEFAULT_MODEL_FILENAME}"

if ENV_MODEL_PATH and os.path.exists(ENV_MODEL_PATH):
    MODEL_PATH = ENV_MODEL_PATH
elif os.path.exists(LOCAL_MODEL_PATH):
    MODEL_PATH = LOCAL_MODEL_PATH
elif os.path.exists(DOCKER_MODEL_PATH):
    MODEL_PATH = DOCKER_MODEL_PATH
else:
    MODEL_PATH = LOCAL_MODEL_PATH

# --- FLASK APP INITIALIZATION ---
app = Flask(__name__, static_folder='.', static_url_path='')
app.secret_key = os.environ.get('SECRET_KEY', 'cybersecuredocai-secret-key-vit-bhopal-2026')
os.makedirs(UPLOAD_FOLDER_BASE, exist_ok=True)
os.makedirs(VECTORSTORE_BASE, exist_ok=True)
os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)

# --- GLOBAL RAG COMPONENTS ---
embeddings = None
llm = None
qa_chain = None
model_load_status = {"status": "uninitialized", "error": None}

# --- AI MODEL INITIALIZATION ---
def initialize_ai_components():
    global embeddings, llm, qa_chain, model_load_status
    print("=" * 60)
    print("Initializing CyberSecureDocAI Offline AI Core...")
    print("=" * 60)

    # 1. Initialize Local Embeddings
    print("[1/2] Loading local embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("  -> Local embedding model loaded successfully.")
    except Exception as e:
        err_msg = f"Failed to initialize embeddings: {e}"
        print(f"CRITICAL: {err_msg}")
        model_load_status = {"status": "error", "error": err_msg}
        return

    # 2. Initialize Local LLM via LlamaCpp
    print(f"[2/2] Loading local quantized LLM from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        err_msg = f"Model file not found at '{MODEL_PATH}'. Please run 'python download_models.py' first."
        print(f"WARNING: {err_msg}")
        model_load_status = {"status": "missing_model", "error": err_msg}
        return

    try:
        # Optimized configuration for CPU AVX2 execution as detailed in Chapter 5
        llm = LlamaCpp(
            model_path=MODEL_PATH,
            n_gpu_layers=0,
            n_ctx=4096,
            temperature=0.2,
            max_tokens=1024,
            repetition_penalty=1.15,
            verbose=False,
            stop=["<|im_end|>", "<|im_start|>"]
        )
        print("  -> Local LLM (Qwen2 1.5B Instruct) initialized successfully.")

        # ChatML format tailored for Qwen2
        prompt_template = """<|im_start|>system
You are CyberSecureDocAI, an offline, privacy-first cybersecurity document analysis assistant. Answer the user's question accurately based ONLY on the provided context excerpts from their private documents. If the context does not contain sufficient information to answer the question, clearly state: "The provided documents do not contain sufficient information on this topic." Do not fabricate answers.<|im_end|>
<|im_start|>user
Context Excerpts:
{context}

User Question:
{input}<|im_end|>
<|im_start|>assistant
"""
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "input"])
        qa_chain = create_stuff_documents_chain(llm, PROMPT)
        model_load_status = {"status": "ready", "error": None}
        print("  -> RAG Stuff Documents Chain ready for offline inference.")

    except Exception as e:
        err_msg = f"Error initializing LlamaCpp: {e}"
        print(f"CRITICAL: {err_msg}")
        traceback.print_exc()
        llm = None
        qa_chain = None
        model_load_status = {"status": "error", "error": err_msg}

# --- DATABASE & PERSISTENCE ---
def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    # Users table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    # Documents table with metadata
    conn.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            file_size INTEGER DEFAULT 0,
            chunk_count INTEGER DEFAULT 0,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    # Chat History table for session memory
    conn.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            message TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    conn.commit()
    conn.close()

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def get_user_folders(user_id: int):
    user_upload_folder = os.path.join(UPLOAD_FOLDER_BASE, f"user_{user_id}")
    user_vectorstore_path = os.path.join(VECTORSTORE_BASE, f"user_{user_id}")
    os.makedirs(user_upload_folder, exist_ok=True)
    return user_upload_folder, user_vectorstore_path

# --- DOCUMENT PARSING & CHUNKING ---
def extract_text_from_file(file_path: str) -> tuple[str, list[dict]]:
    """Extracts text and page metadata from PDF or plain text files."""
    text_content = ""
    metadata_list = []
    filename = os.path.basename(file_path)
    ext = os.path.splitext(filename)[1].lower()

    if ext == '.pdf':
        try:
            reader = PdfReader(file_path, strict=False)
            for idx, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                clean_text = page_text.encode('ascii', 'ignore').decode('ascii').strip()
                if clean_text:
                    text_content += f"\n--- [{filename} Page {idx + 1}] ---\n{clean_text}\n"
                    metadata_list.append({"page": idx + 1, "text": clean_text})
        except Exception as e:
            print(f"Error parsing PDF {filename}: {e}")
    elif ext in ['.txt', '.md', '.log', '.json']:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                text_content += f"\n--- [{filename}] ---\n{content}\n"
                metadata_list.append({"page": 1, "text": content})
        except Exception as e:
            print(f"Error reading file {filename}: {e}")
    return text_content, metadata_list

def get_text_chunks(text: str):
    """Splits document text into overlapping chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_text(text)

def rebuild_user_vectorstore(user_id: int) -> tuple[bool, int]:
    """Rebuilds the FAISS vector index for all documents owned by user_id."""
    if not embeddings:
        return False, 0
    
    user_upload_folder, user_vectorstore_path = get_user_folders(user_id)
    conn = get_db_connection()
    docs_cursor = conn.execute('SELECT id, filename FROM documents WHERE user_id = ?', (user_id,))
    user_docs = docs_cursor.fetchall()

    if not user_docs:
        if os.path.exists(user_vectorstore_path):
            shutil.rmtree(user_vectorstore_path)
        conn.close()
        return True, 0

    all_chunks = []
    metadatas = []
    
    for row in user_docs:
        doc_id = row['id']
        filename = row['filename']
        file_path = os.path.join(user_upload_folder, filename)
        if os.path.exists(file_path):
            file_text, _ = extract_text_from_file(file_path)
            chunks = get_text_chunks(file_text)
            conn.execute('UPDATE documents SET chunk_count = ? WHERE id = ?', (len(chunks), doc_id))
            for chunk in chunks:
                all_chunks.append(chunk)
                metadatas.append({"source": filename})
    
    conn.commit()
    conn.close()

    if not all_chunks:
        if os.path.exists(user_vectorstore_path):
            shutil.rmtree(user_vectorstore_path)
        return True, 0

    try:
        new_vector_store = FAISS.from_texts(
            texts=all_chunks,
            embedding=embeddings,
            metadatas=metadatas
        )
        if os.path.exists(user_vectorstore_path):
            shutil.rmtree(user_vectorstore_path)
        new_vector_store.save_local(user_vectorstore_path)
        return True, len(all_chunks)
    except Exception as e:
        print(f"Error rebuilding vectorstore for user {user_id}: {e}")
        traceback.print_exc()
        return False, 0

# --- REST API ROUTES ---
@app.route('/')
def root():
    return send_from_directory('.', 'index.html')

@app.route('/api/status', methods=['GET'])
def system_status():
    """System health check and offline status."""
    return jsonify({
        "system": "CyberSecureDocAI",
        "mode": "100% Offline / Air-Gapped",
        "model_path": MODEL_PATH,
        "model_loaded": llm is not None,
        "embeddings_loaded": embeddings is not None,
        "model_status": model_load_status["status"],
        "model_error": model_load_status["error"]
    })

@app.route('/api/register', methods=['POST'])
def register():
    data = request.json or {}
    username = data.get('username', '').strip()
    email = data.get('email', '').strip().lower()
    password = data.get('password', '').strip()

    if not username or not email or not password:
        return jsonify({"message": "Please fill in all registration fields."}), 400
    if len(password) < 6:
        return jsonify({"message": "Password must be at least 6 characters long."}), 400

    password_hash = hash_password(password)
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (username, email, password_hash)
        )
        conn.commit()
        user_id = cursor.lastrowid
        session['user_id'] = user_id
        session['username'] = username
        return jsonify({
            "message": "Account created successfully.",
            "user": {"id": user_id, "username": username, "email": email}
        }), 201
    except sqlite3.IntegrityError:
        return jsonify({"message": "A user with that username or email already exists."}), 409
    finally:
        conn.close()

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json or {}
    email_or_username = data.get('email', '').strip()
    password = data.get('password', '').strip()

    if not email_or_username or not password:
        return jsonify({"message": "Please enter your username/email and password."}), 400

    password_hash = hash_password(password)
    conn = get_db_connection()
    user = conn.execute(
        'SELECT * FROM users WHERE (email = ? OR username = ?) AND password_hash = ?',
        (email_or_username.lower(), email_or_username, password_hash)
    ).fetchone()
    conn.close()

    if user:
        session['user_id'] = user['id']
        session['username'] = user['username']
        return jsonify({
            "message": "Welcome back! Login successful.",
            "user": {"id": user['id'], "username": user['username'], "email": user['email']}
        })
    return jsonify({"message": "Invalid username/email or password."}), 401

@app.route('/api/logout', methods=['GET', 'POST'])
def logout():
    session.clear()
    return jsonify({"message": "Logged out successfully."})

@app.route('/api/check_session', methods=['GET'])
def check_session():
    if 'user_id' in session:
        conn = get_db_connection()
        user = conn.execute('SELECT id, username, email FROM users WHERE id = ?', (session['user_id'],)).fetchone()
        conn.close()
        if user:
            return jsonify({
                "logged_in": True,
                "user": {"id": user['id'], "username": user['username'], "email": user['email']}
            })
    return jsonify({"logged_in": False})

@app.before_request
def require_login():
    public_endpoints = ['root', 'static', 'login', 'register', 'check_session', 'system_status']
    if request.endpoint and request.endpoint not in public_endpoints and 'user_id' not in session:
        return jsonify({"message": "Authentication required. Please log in."}), 401

@app.route('/api/upload', methods=['POST'])
def upload_files():
    user_id = session['user_id']
    user_upload_folder, _ = get_user_folders(user_id)
    files = request.files.getlist('files')

    if not files or files[0].filename == '':
        return jsonify({'success': False, 'message': 'No files selected for upload.'}), 400

    conn = get_db_connection()
    uploaded_count = 0
    allowed_extensions = {'.pdf', '.txt', '.md', '.log'}

    for file in files:
        if file and file.filename:
            raw_filename = secure_filename(file.filename)
            ext = os.path.splitext(raw_filename)[1].lower()
            if ext not in allowed_extensions:
                continue

            dest_path = os.path.join(user_upload_folder, raw_filename)
            file.save(dest_path)
            file_size = os.path.getsize(dest_path)

            existing = conn.execute(
                'SELECT id FROM documents WHERE user_id = ? AND filename = ?',
                (user_id, raw_filename)
            ).fetchone()

            if not existing:
                conn.execute(
                    'INSERT INTO documents (user_id, filename, file_size) VALUES (?, ?, ?)',
                    (user_id, raw_filename, file_size)
                )
            else:
                conn.execute(
                    'UPDATE documents SET file_size = ? WHERE id = ?',
                    (file_size, existing['id'])
                )
            uploaded_count += 1

    conn.commit()
    conn.close()

    if uploaded_count == 0:
        return jsonify({'success': False, 'message': 'No supported files (.pdf, .txt, .md) found.'}), 400

    success, chunk_count = rebuild_user_vectorstore(user_id)
    if not success:
        return jsonify({'success': False, 'message': 'Uploaded files, but vector indexing failed. Ensure embedding model is ready.'}), 500

    return jsonify({
        'success': True,
        'message': f'Successfully ingested {uploaded_count} file(s) into {chunk_count} vector chunks.'
    })

@app.route('/api/documents', methods=['GET'])
def get_documents():
    user_id = session['user_id']
    conn = get_db_connection()
    cursor = conn.execute(
        'SELECT id, filename, file_size, chunk_count, uploaded_at FROM documents WHERE user_id = ? ORDER BY id DESC',
        (user_id,)
    )
    docs = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return jsonify({'documents': docs})

@app.route('/api/delete/<path:filename>', methods=['DELETE'])
def delete_document(filename):
    user_id = session['user_id']
    user_upload_folder, _ = get_user_folders(user_id)
    secure_name = secure_filename(filename)

    conn = get_db_connection()
    conn.execute('DELETE FROM documents WHERE user_id = ? AND filename = ?', (user_id, secure_name))
    conn.commit()
    conn.close()

    filepath = os.path.join(user_upload_folder, secure_name)
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
        except Exception as e:
            print(f"Warning: could not delete file {filepath}: {e}")

    rebuild_user_vectorstore(user_id)
    return jsonify({'success': True, 'message': f'Document "{secure_name}" removed from knowledge base.'})

@app.route('/api/chat', methods=['POST'])
def chat():
    user_id = session['user_id']
    _, user_vectorstore_path = get_user_folders(user_id)
    payload = request.json or {}
    user_message = payload.get('message', '').strip()

    if not user_message:
        return jsonify({'reply': 'Please enter a question or query.'}), 400

    if not qa_chain:
        if model_load_status["status"] == "missing_model":
            return jsonify({
                'reply': 'The offline LLM model is not found. Please run "python download_models.py" to download Qwen2-1.5B weights.'
            }), 503
        return jsonify({
            'reply': f'AI generation engine not available ({model_load_status.get("error") or "Still initializing"}).'
        }), 503

    if not os.path.exists(user_vectorstore_path):
        return jsonify({
            'reply': 'No document vector index found. Please upload at least one PDF or document first to analyze.'
        }), 400

    start_time = time.time()
    try:
        # Load user-specific isolated FAISS vector store
        current_vector_store = FAISS.load_local(
            user_vectorstore_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

        # 1. Similarity Retrieval
        retrieved_docs = current_vector_store.similarity_search(user_message, k=3)
        if not retrieved_docs:
            return jsonify({
                'reply': "Could not find any relevant information in your uploaded documents.",
                'sources': []
            })

        # 2. Extract citations
        sources = []
        for d in retrieved_docs:
            source_file = d.metadata.get("source", "Document")
            snippet = d.page_content.strip()
            if len(snippet) > 280:
                snippet = snippet[:280] + "..."
            sources.append({"source": source_file, "snippet": snippet})

        # 3. Augmentation & Generation
        raw_response = qa_chain.invoke({"input": user_message, "context": retrieved_docs})
        reply_text = raw_response.strip()

        # Clean any trailing prompt artifact if present
        if "<|im_end|>" in reply_text:
            reply_text = reply_text.split("<|im_end|>")[0].strip()

        elapsed_sec = round(time.time() - start_time, 2)

        # Persist conversation to chat history
        conn = get_db_connection()
        conn.execute('INSERT INTO chat_history (user_id, role, message) VALUES (?, ?, ?)', (user_id, 'user', user_message))
        conn.execute('INSERT INTO chat_history (user_id, role, message) VALUES (?, ?, ?)', (user_id, 'assistant', reply_text))
        conn.commit()
        conn.close()

        return jsonify({
            'reply': reply_text,
            'sources': sources,
            'latency': f"{elapsed_sec}s"
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'reply': f'An error occurred during inference: {str(e)}',
            'sources': []
        }), 500

@app.route('/api/chat/history', methods=['GET'])
def get_chat_history():
    user_id = session['user_id']
    conn = get_db_connection()
    cursor = conn.execute(
        'SELECT role, message, timestamp FROM chat_history WHERE user_id = ? ORDER BY id ASC LIMIT 50',
        (user_id,)
    )
    history = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return jsonify({'history': history})

@app.route('/api/chat/history', methods=['DELETE'])
def clear_chat_history():
    user_id = session['user_id']
    conn = get_db_connection()
    conn.execute('DELETE FROM chat_history WHERE user_id = ?', (user_id,))
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'message': 'Chat history cleared.'})

if __name__ == '__main__':
    print("=" * 60)
    print(" starting CyberSecureDocAI Application Server...")
    print("=" * 60)
    init_db()
    initialize_ai_components()
    print("\nServer running at: http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
