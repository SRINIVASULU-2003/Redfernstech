from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

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

    # Create a generic response
    return make_response(f"Response for intent: {intent}")

def make_response(message):
    return {
        'fulfillmentText': message
    }


