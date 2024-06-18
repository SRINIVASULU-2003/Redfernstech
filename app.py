from flask import Flask, request, render_template, jsonify
import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

app = Flask(__name__)

# Set your HuggingFace API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=128, temperature=0.7)

# Load PDF documents from the specified directory
loader = PyPDFDirectoryLoader("C:\\Users\\USER\\OneDrive - Lakireddy Bali Reddy College of Engineering\\Documents\\Redfernstech\\pdfs")  # Update with your directory path
docs = loader.load()
print(f"Number of documents loaded: {len(docs)}")

# Instantiate embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
all_splits = text_splitter.split_documents(docs)

# Create Qdrant collection
qdrant_collection = Qdrant.from_documents(all_splits, embeddings, location=":memory:", collection_name="all_documents")

# Create retriever
retriever = qdrant_collection.as_retriever()

# Create RetrievalQA object
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

def get_response_from_huggingface(prompt):
    complete_prompt = (
        f"You are the Redfernstech chatbot. Please provide your answers using the "
        f"information below in bullet points. Ensure the response is between 40 to 60 words.\n\n"
        f"Query: {prompt}\n\n"
        f"Response:"
    )
    response = qa.invoke(complete_prompt)['result']
    return response

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    bot_response = get_response_from_huggingface(user_message)
    return jsonify({"response": bot_response})

@app.route('/webhook', methods=['POST'])
def webhook():
    req = request.get_json(silent=True, force=True)
    user_message = req.get('queryResult', {}).get('queryText', '')

    # Pass the user's message directly to the model
    hf_response = get_response_from_huggingface(user_message)

    return make_response(hf_response)

def make_response(message):
    return {
        'fulfillmentText': message
    }

