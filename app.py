from flask import Flask, render_template, request, jsonify
import asyncio
from main import generate_response, intent_clf, memory,speak

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    intent, confidence = intent_clf.predict_intent(user_input)
    response = generate_response(user_input, intent, confidence)

    # Optional: trigger text-to-speech (speak in background)
    asyncio.run(speak(response))

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
