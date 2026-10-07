from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from google import genai
import os

load_dotenv()

app = Flask(__name__)

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing in .env")

client = genai.Client(api_key=API_KEY)


SYSTEM_PROMPT = """
You are CodeForge Voice AI.

LANGUAGE RULE — VERY IMPORTANT:

Detect the language of the user's question.

If the user asks in Tamil:
- Answer ONLY in Tamil.
- Do NOT translate the answer into English.
- Do NOT use English unnecessarily.
- Explain naturally in spoken Tamil.

If the user asks in Tanglish:
- Answer in simple Tanglish.

If the user asks in English:
- Answer in English.

If the user asks in Hindi:
- Answer in Hindi.

If the user asks in Marathi:
- Answer in Marathi.

Always answer in the SAME LANGUAGE used by the user.

Examples:

User: Python என்றால் என்ன?
Answer: Python என்பது ஒரு programming language. இது web development, artificial intelligence, machine learning போன்றவற்றில் பயன்படுத்தப்படுகிறது.

User: AI na enna?
Answer: AI என்பது Artificial Intelligence. இது மனிதர்களைப் போல தகவல்களை புரிந்து கொண்டு முடிவெடுக்க உதவும் தொழில்நுட்பம்.

User: What is Python?
Answer: Python is a high-level programming language.

Keep answers clear, natural and easy to understand.
"""


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():

    data = request.get_json(silent=True) or {}

    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({
            "answer": "தயவுசெய்து ஒரு கேள்வியை கேளுங்கள்."
        }), 400

    try:

        response = client.models.generate_content(
            model=MODEL,
            contents=question,
            config={
                "system_instruction": SYSTEM_PROMPT,
                "temperature": 0.2
            }
        )

        answer = response.text or "மன்னிக்கவும், பதில் உருவாக்க முடியவில்லை."

        return jsonify({
            "answer": answer
        })

    except Exception as e:

        return jsonify({
            "answer": f"AI Error: {str(e)}"
        }), 500


@app.route("/health")
def health():

    return jsonify({
        "status": "CodeForge Voice AI is running"
    })


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )