import os
from flask import Flask, request, jsonify
from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex, SimpleDirectoryReader, ChatPromptTemplate, Settings
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF

app = Flask(__name__)

# Configure the Llama index settings
Settings.llm = HuggingFaceInferenceAPI(
    model_name="google/gemma-1.1-7b-it",
    tokenizer_name="google/gemma-1.1-7b-it",
    context_window=3000,
    token="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",
    max_new_tokens=512,
    generate_kwargs={"temperature": 0.1},
    model_config={'protected_namespaces': ()}  # To resolve the UserWarning
)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# Define the directory for persistent storage and data
PERSIST_DIR = "db"
UPLOAD_FOLDER = 'uploads'

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)

def extract_text_from_pdf(file_path):
    text = ""
    doc = fitz.open(file_path)
    for page in doc:
        text += page.get_text()
    return text

def data_ingestion(file_path):
    text = extract_text_from_pdf(file_path)
    documents = [text]
    storage_context = StorageContext.from_defaults()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)

def handle_query(query):
    chat_text_qa_msgs = [
        (
            "user",
            """You are a Q&A assistant named RedfernsTech, created by the RedfernsTech team. You have a specific response programmed for when users specifically ask about your creator. The response is: "I was created by RedfernsTech, a team passionate about Artificial Intelligence and technology. We are dedicated to providing the best user experiences by solving complex problems and delivering innovative solutions. With expertise in machine learning, deep learning, Python, generative AI, NLP, and computer vision, RedfernsTech aims to push the boundaries of AI to create new possibilities." For all other inquiries, your main goal is to provide answers as accurately as possible, based on the instructions and context you have been given. If a question does not match the provided context or is outside the scope of the document, kindly advise the user to ask questions within the context of the document.
            Context:
            {context_str}
            Question:
            {query_str}
            """

        )
    ]
    text_qa_template = ChatPromptTemplate.from_messages(chat_text_qa_msgs)

    # Load index from storage
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

    query_engine = index.as_query_engine(text_qa_template=text_qa_template)
    answer = query_engine.query(query)

    if hasattr(answer, 'response'):
        return answer.response
    elif isinstance(answer, dict) and 'response' in answer:
        return answer['response']
    else:
        return "Sorry, I couldn't find an answer."

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    files = request.files.getlist('file')
    file_paths = []
    for file in files:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        file_paths.append(file_path)
        data_ingestion(file_path)
    
    return jsonify({"message": "Files successfully uploaded and ingested"}), 200

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "No query provided"}), 400
    
    query = data['query']
    answer = handle_query(query)
    return jsonify({"answer": answer}), 200

