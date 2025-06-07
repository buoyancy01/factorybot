from flask import Flask, request, jsonify, render_template_string
import requests, os, openai
from collections import defaultdict
from bs4 import BeautifulSoup
import faiss
import pickle
import time
import hashlib

app = Flask(__name__)

# ✅ ENV VARS
TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH")
TWILIO_NUMBER = 'whatsapp:+14155238886'
SHIP24_API_KEY = os.getenv("SHIP24_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# ✅ Ensure dirs exist
os.makedirs("logs", exist_ok=True)
os.makedirs("vectordb", exist_ok=True)

# ✅ MEMORY STORAGE
user_memory = defaultdict(list)

@app.route('/whatsapp', methods=['POST'])
def whatsapp():
    msg = request.form.get('Body', '').strip()
    sender = request.form.get('From')

    response_text = None

    if msg.lower().startswith("track "):
        tracking_number = msg.split("track ", 1)[1].strip()
        response_text = track_package(tracking_number)

    elif msg.lower().startswith("product "):
        keyword = msg.split("product ", 1)[1].strip()
        response_text = vector_search_product(keyword)

    elif msg.lower() == "form":
        response_text = form_reply()

    else:
        response_text = gpt_reply(sender, msg)

    send_whatsapp_message(sender, response_text)
    return "OK", 200


def send_whatsapp_message(to, body):
    url = f'https://api.twilio.com/2010-04-01/Accounts/{TWILIO_SID}/Messages.json'
    auth = (TWILIO_SID, TWILIO_AUTH)
    data = {
        'From': TWILIO_NUMBER,
        'To': to,
        'Body': body
    }
    requests.post(url, data=data, auth=auth)


def track_package(tracking_number):
    url = f"https://api.ship24.com/public/v1/trackers/{tracking_number}"
    headers = {'Authorization': f'Bearer {SHIP24_API_KEY}'}
    res = requests.get(url, headers=headers).json()

    try:
        event = res['data']['trackers'][0]['events'][0]
        return f"📦 {event['description']} at {event['location']} on {event['datetime']}"
    except:
        return "❌ Couldn't find tracking info. Please double-check your number."


def form_reply():
    return "📝 Latest form response: Name: John Doe, Email: john@example.com"


def gpt_reply(user_id, user_msg):
    save_convo(user_id, 'user', user_msg)
    history = user_memory[user_id][-6:]
    prompt = [
        {"role": "system", "content": "You are a friendly customer service assistant for an online store."}
    ] + history

    try:
        res = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=prompt,
            temperature=0.6,
            max_tokens=150
        )
        reply = res['choices'][0]['message']['content'].strip()
        save_convo(user_id, 'assistant', reply)
        return reply
    except Exception as e:
        return "⚠️ GPT failed. Please try again."


def save_convo(user_id, role, content):
    user_memory[user_id].append({"role": role, "content": content})
    with open(f"logs/{hashlib.sha256(user_id.encode()).hexdigest()}.txt", "a", encoding="utf-8") as f:
        f.write(f"{role}: {content}\n")


@app.route('/refresh_products')
def refresh_products():
    try:
        html = requests.get("https://shop.factory43.com").text
        soup = BeautifulSoup(html, 'html.parser')
        products = [tag.get_text(strip=True) for tag in soup.find_all(['h1', 'h2', 'h3', 'p', 'a'])]
        chunks = [p for p in products if len(p) > 20]

        vectors, texts = [], []
        for chunk in chunks:
            emb = openai.Embedding.create(input=chunk, model="text-embedding-ada-002")
            vectors.append(emb['data'][0]['embedding'])
            texts.append(chunk)

        index = faiss.IndexFlatL2(len(vectors[0]))
        index.add(np.array(vectors).astype('float32'))

        with open("vectordb/index.pkl", "wb") as f:
            pickle.dump((index, texts), f)

        return "✅ Vector DB refreshed with product data."
    except Exception as e:
        return f"❌ Refresh failed: {e}"


def vector_search_product(query):
    try:
        with open("vectordb/index.pkl", "rb") as f:
            index, texts = pickle.load(f)
        emb = openai.Embedding.create(input=query, model="text-embedding-ada-002")
        D, I = index.search(np.array([emb['data'][0]['embedding']]).astype('float32'), 1)
        return f"🔍 Closest match: {texts[I[0][0]]}"
    except:
        return "⚠️ Product search failed. Try again later."


@app.route('/admin/logs')
def admin_logs():
    entries = []
    for fname in os.listdir("logs"):
        with open(f"logs/{fname}", encoding='utf-8') as f:
            entries.append((fname, f.read()))

    html = """
    <html><head><title>Admin Logs</title>
    <style>
    body { font-family: 'Segoe UI', sans-serif; background: #f5f7fa; padding: 20px; }
    h2 { color: #333; }
    .log { background: white; box-shadow: 0 0 10px rgba(0,0,0,0.1); padding: 15px; margin-bottom: 20px; border-radius: 8px; }
    pre { white-space: pre-wrap; word-wrap: break-word; }
    </style></head><body>
    <h2>📊 WhatsApp User Logs</h2>
    {% for fname, content in entries %}
        <div class="log">
            <strong>{{ fname }}</strong>
            <pre>{{ content }}</pre>
        </div>
    {% endfor %}
    </body></html>
    """
    return render_template_string(html, entries=entries)
