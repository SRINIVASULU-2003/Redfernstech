import os
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

HUGGINGFACE_API_KEY = "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"
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/Mistal7B"

@app.route('/')
def hello_world():
    return 'Hello, World! RedFerns'

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
    hf_response = call_huggingface_api(intent)

    # Create a response based on the Hugging Face API result
    return make_response(hf_response)

def call_huggingface_api(prompt):
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_KEY}"
    }
    data = {
        "inputs": prompt
    }

    response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        return result.get("generated_text", "No response generated.")
    else:
        print(f"Error {response.status_code}: {response.text}")
        return "Error processing the request."

def make_response(message):
    return {
        'fulfillmentText': message
    }


