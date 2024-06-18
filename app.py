from flask import Flask, request, render_template, jsonify
import os
from langchain_huggingface import HuggingFaceEndpoint

app = Flask(__name__)

# Set your HuggingFace API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=128, temperature=0.7)

def get_response_from_huggingface(prompt):
    response = llm.invoke(prompt)
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
    res = process_request(req)
    return jsonify(res)

def process_request(req):
    # Extract the intent name from the request for logging or further use
    intent = req.get('queryResult').get('intent').get('displayName')

    # Log the received intent
    print(f"Received intent: {intent}")

    # Call the Hugging Face API with the intent
    hf_response = get_response_from_huggingface(intent)

    # Create a response based on the Hugging Face API result
    return make_response(hf_response)

def make_response(message):
    return {
        'fulfillmentText': message
    }

