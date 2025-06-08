from flask import Flask, request, render_template_string
import requests, os, openai, json
from collections import defaultdict
import datetime
import numpy as np
import faiss
from bs4 import BeautifulSoup

app = Flask(__name__)

# ✅ ENV VARS
TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH")
TWILIO_NUMBER = 'whatsapp:+14155238886'
SHIP24_API_KEY = os.getenv("SHIP24_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai.api_key = OPENAI_API_KEY

# ✅ MEMORY STORAGE
MEMORY_DIR = "logs"
VECTOR_DIR = "vectordb"
os.makedirs(MEMORY_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)
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
        response_text = search_product(keyword)

    elif msg.lower() == "form":
        response_text = form_reply()

    else:
        response_text = gpt_reply(sender, msg)

    send_whatsapp_message(sender, response_text)
    return "OK", 200

@app.route("/admin/logs")
def admin_logs():
    logs_html = "<h1>📊 WhatsApp User Logs</h1><ul>"
    for filename in sorted(os.listdir(MEMORY_DIR)):
        with open(os.path.join(MEMORY_DIR, filename)) as f:
            convo = json.load(f)
        logs_html += f"<li><strong>{filename}</strong><ul>"
        for entry in convo:
            logs_html += f"<li><b>{entry['role']}:</b> {entry['content']}</li>"
        logs_html += "</ul></li>"
    logs_html += "</ul>"
    return render_template_string(logs_html)

@app.route("/refresh_products")
def refresh_products():
    try:
        url = "https://shop.factory43.com/"
        soup = BeautifulSoup(requests.get(url).text, "html.parser")
        text = soup.get_text()

        chunks = [text[i:i+500] for i in range(0, len(text), 500)]
        embeddings = []
        for chunk in chunks:
            res = openai.embeddings.create(input=chunk, model="text-embedding-ada-002")
            embeddings.append(np.array(res.data[0].embedding, dtype=np.float32))

        index = faiss.IndexFlatL2(len(embeddings[0]))
        index.add(np.array(embeddings))

        faiss.write_index(index, os.path.join(VECTOR_DIR, "product.idx"))
        with open(os.path.join(VECTOR_DIR, "chunks.json"), "w") as f:
            json.dump(chunks, f)

        return "✅ Vector DB refreshed."
    except Exception as e:
        return f"❌ Refresh failed: {e}"

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

def search_product(keyword):
    try:
        index_path = os.path.join(VECTOR_DIR, "product.idx")
        chunks_path = os.path.join(VECTOR_DIR, "chunks.json")
        if not os.path.exists(index_path) or not os.path.exists(chunks_path):
            return "⚠️ Vector DB not built yet. Please refresh."

        index = faiss.read_index(index_path)
        with open(chunks_path) as f:
            chunks = json.load(f)

        res = openai.embeddings.create(input=keyword, model="text-embedding-ada-002")
        query = np.array(res.data[0].embedding, dtype=np.float32)

        D, I = index.search(np.array([query]), k=1)
        match = chunks[I[0][0]]

        return f"🔍 Top result for '{keyword}':\n" + match.strip()[:300]
    except Exception as e:
        return f"❌ Search failed: {e}"

def form_reply():
    return "📝 Latest form response: Name: John Doe, Email: john@example.com"

def gpt_reply(user_id, user_msg):
    save_convo(user_id, 'user', user_msg)

    history = user_memory[user_id][-6:]
    prompt = [{"role": "system", "content": "You are a helpful assistant for an e-commerce store."}] + history

    try:
        res = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=prompt,
            temperature=0.6,
            max_tokens=150
        )
        reply = res.choices[0].message.content.strip()
        save_convo(user_id, 'assistant', reply)
        return reply
    except Exception as e:
        return f"⚠️ GPT failed: {e}"

def save_convo(user_id, role, content):
    user_memory[user_id].append({"role": role, "content": content})
    filepath = os.path.join(MEMORY_DIR, user_id.replace(":", "") + ".json")
    with open(filepath, "w") as f:
        json.dump(user_memory[user_id], f)
